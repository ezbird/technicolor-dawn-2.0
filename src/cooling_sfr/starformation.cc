/*! \file starformation.cc
 *
 *  \brief Implements star formation and related routines based on the effective multi-phase model
 */

#include "gadgetconfig.h"

#ifdef COOLING
#ifdef STARFORMATION

#include <gsl/gsl_math.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/intposconvert.h"
#include "../data/mymalloc.h"
#include "../logs/logs.h"
#include "../logs/timer.h"
#include "../system/system.h"
#include "../time_integration/timestep.h"
#include "../cooling_sfr/cooling.h"

 // Debug output function
 #define STARFORMATION_PRINT(...) \
     do { if (All.StarFormationDebugLevel) printf("[STARFORMATION] " __VA_ARGS__); } while (0)


/*! \brief Finalize star formation module
 *
 *  This function closes the star formation log file
 *  when the simulation ends.
 */
void end_starformation(void)
{
  TIMER_START(CPU_MISC);

  printf("STARFORMATION: Ending star formation module...\n");

  TIMER_STOP(CPU_MISC);
}

/**
 * Set up units for the star formation code
 */
void coolsfr::set_units_sfr(void)
{
  double meanweight;

#ifdef COSMIC_RAYS
  double feedbackenergyinergs;
#endif

  All.OverDensThresh =
    All.CritOverDensity * All.OmegaBaryon * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G);

  All.PhysDensThresh = All.CritPhysDensity * PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs;

  meanweight = 4 / (1 + 3 * HYDROGEN_MASSFRAC);	/* assuming NEUTRAL GAS */

  All.EgySpecCold = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.TempClouds;
  All.EgySpecCold *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

  meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));	/* assuming FULL ionization */

  All.EgySpecSN = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.TempSupernova;
  All.EgySpecSN *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

#ifdef COSMIC_RAYS
  if(All.CR_SNEff < 0.0)
    {
      /* if CR_SNeff < 0.0, then subtract CR Feedback energy from thermal feedback energy */
      if(ThisTask == 0)
        {
          mpi_printf("COOLSFR: %g percent of thermal feedback go into Cosmic Rays.\n", -100.0 * All.CR_SNEff);
        }

      All.EgySpecSN *= (1.0 + All.CR_SNEff);
      All.CR_SNEff = -All.CR_SNEff / (1.0 + All.CR_SNEff);
    }

  All.FeedbackEnergy = All.FactorSN / (1 - All.FactorSN) * All.EgySpecSN;

  feedbackenergyinergs = All.FeedbackEnergy / All.UnitMass_in_g * (All.UnitEnergy_in_cgs * SOLAR_MASS);
#endif

  if(ThisTask == 0)
    {
#ifdef COSMIC_RAYS
      mpi_printf("COOLSFR: Feedback energy per formed solar mass in stars= %g ergs\n", feedbackenergyinergs);
#endif
      mpi_printf("COOLSFR: OverDensThresh= %g\nPhysDensThresh= %g (internal units)\n", All.OverDensThresh, All.PhysDensThresh);
    }
}

/**
 * Initialize the star formation module
 * This sets up the tables for pressures and other physical parameters
 */
void coolsfr::init_clouds(void)
{
  double A0, dens, tcool, ne, coolrate, egyhot, x, u4, meanweight;
  double tsfr, y, peff, fac, neff, egyeff, factorEVP, sigma, thresholdStarburst;
  
  if(All.PhysDensThresh == 0)
    {
      A0 = All.FactorEVP;

      egyhot = All.EgySpecSN / A0;

      meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));	/* assuming FULL ionization */

      u4 = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * 1.0e4;
      u4 *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

      if(All.ComovingIntegrationOn)
        dens = 1.0e6 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G);
      else
        // Physical cgs value, then convert to system units
        dens = 9.205e-24 / (All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam);

      // Save current time state
      double oldtime = All.Time;
      
      if(All.ComovingIntegrationOn)
        {
          All.Time = 1.0;	/* to be guaranteed to get z=0 rate */
          IonizeParams();
        }

      ne = 1.0;
      SetZeroIonization();
      
      gas_state gs = GasState;
      do_cool_data DoCool;
      
#ifdef TABLECOOL
#ifdef METALCOOL
      tcool = GetCoolingTime(egyhot, dens, &ne, &gs, &DoCool); // Use general cooling functions
#else
      tcool = GetCoolingTime(egyhot, dens, &ne, &gs, &DoCool);
#endif
#else
      tcool = GetCoolingTime(egyhot, dens, &ne, &gs, &DoCool);
#endif

      coolrate = egyhot / tcool / dens;

      x = (egyhot - u4) / (egyhot - All.EgySpecCold);

      All.PhysDensThresh =
        x / pow(1 - x, 2) * (All.FactorSN * All.EgySpecSN - (1 -
                                                             All.FactorSN) * All.EgySpecCold) /
        (All.MaxSfrTimescale * coolrate);

      if(ThisTask == 0)
        {
          mpi_printf("\nCOOLSFR: A0= %g\n", A0);
          mpi_printf("COOLSFR: Computed: PhysDensThresh= %g (int units) %g h^2 cm^-3\n", All.PhysDensThresh,
                     All.PhysDensThresh / (PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs));
          mpi_printf("COOLSFR: EXPECTED FRACTION OF COLD GAS AT THRESHOLD = %g\n\n", x);
          mpi_printf("COOLSFR: tcool=%g dens=%g egyhot=%g\n", tcool, dens, egyhot);
        }

      dens = All.PhysDensThresh * 10;

      do
        {
          tsfr = sqrt(All.PhysDensThresh / (dens)) * All.MaxSfrTimescale;
          factorEVP = pow(dens / All.PhysDensThresh, -0.8) * All.FactorEVP;
          egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

          ne = 0.5;
#ifdef TABLECOOL
#ifdef METALCOOL
          tcool = GetCoolingTime(egyhot, dens, &ne, &gs, &DoCool);
#else
          tcool = GetCoolingTime(egyhot, dens, &ne, &gs, &DoCool);
#endif
#else
          tcool = GetCoolingTime(egyhot, dens, &ne, &gs, &DoCool);
#endif

          y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
          x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));
          egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

          peff = GAMMA_MINUS1 * dens * egyeff;

          fac = 1 / (log(dens * 1.025) - log(dens));
          dens *= 1.025;

          neff = -log(peff) * fac;

          tsfr = sqrt(All.PhysDensThresh / (dens)) * All.MaxSfrTimescale;
          factorEVP = pow(dens / All.PhysDensThresh, -0.8) * All.FactorEVP;
          egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

          ne = 0.5;
#ifdef TABLECOOL
#ifdef METALCOOL
          tcool = GetCoolingTime(egyhot, dens, &ne, &gs, &DoCool);
#else
          tcool = GetCoolingTime(egyhot, dens, &ne, &gs, &DoCool);
#endif
#else
          tcool = GetCoolingTime(egyhot, dens, &ne, &gs, &DoCool);
#endif

          y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
          x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));
          egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

          peff = GAMMA_MINUS1 * dens * egyeff;

          neff += log(peff) * fac;
        }
      while(neff > 4.0 / 3);

      thresholdStarburst = dens;

#ifdef MODIFIEDBONDI
      All.BlackHoleRefDensity = thresholdStarburst;
      All.BlackHoleRefSoundspeed = sqrt(GAMMA * GAMMA_MINUS1 * egyeff);
#endif

      if(ThisTask == 0)
        {
          mpi_printf("COOLSFR: Run-away sets in for dens=%g\n", thresholdStarburst);
          mpi_printf("COOLSFR: Dynamic range for quiescent star formation= %g\n", thresholdStarburst / All.PhysDensThresh);
          fflush(stdout);
        }

      integrate_sfr();

      if(ThisTask == 0)
        {
          sigma = 10.0 / All.Hubble * 1.0e-10 / pow(1.0e-3, 2);

          mpi_printf("COOLSFR: Isotherm sheet central density: %g z0=%g\n",
                     M_PI * All.G * sigma * sigma / (2 * GAMMA_MINUS1) / u4,
                     GAMMA_MINUS1 * u4 / (2 * M_PI * All.G * sigma));
          fflush(stdout);
        }

      // Restore time setting
      if(All.ComovingIntegrationOn)
        {
          All.Time = oldtime;
          IonizeParams();
        }

#ifdef WINDS
      if(All.WindEfficiency > 0)
        if(ThisTask == 0)
          mpi_printf("COOLSFR: Windspeed: %g\n",
                     sqrt(2 * All.WindEnergyFraction * All.FactorSN * All.EgySpecSN / (1 - All.FactorSN) / All.WindEfficiency));
#endif
    }
}

/**
 * Calculate the star formation rate of a gas particle with Gadget-3 style approach
 * Now includes additional temperature check to prevent hot gas from forming stars
 */
double coolsfr::get_starformation_rate(int i, double *xcloud, simparticles *Sp)
{
  double rateOfSF;
  double tsfr, cloudmass;
  gas_state gs = GasState;

  double factorEVP, egyhot, ne, tcool, y, x;
  do_cool_data DoCool = DoCoolData;

  *xcloud = 0.0; /* default: no molecular clouds */

  if(Sp->SphP[i].SfFlag == 0)
    return 0;

  // Perform additional temperature check to ensure the particle is still eligible
  double utherm = Sp->get_utherm_from_entropy(i);
  double rho = Sp->SphP[i].Density * All.cf_a3inv;
  double currentTemp = convert_u_to_temp(utherm, rho, &Sp->SphP[i].Ne, &gs, &DoCool);
  
  // Skip star formation if temperature exceeds threshold
  if(currentTemp > All.MaxStarFormationTemp) {
    if(ThisTask == 0 && All.StarFormationDebugLevel > 1) {
      printf("[SF_DEBUG] Particle %d too hot for SF: T=%g K > %g K\n", 
             Sp->P[i].ID.get(), currentTemp, All.MaxStarFormationTemp);
    }
    // Reset the SF flag to prevent further attempts
    Sp->SphP[i].SfFlag = 0;
    return 0;
  }

  // Using approach similar to Gadget-3's non-H2 model
  tsfr = sqrt(All.PhysDensThresh / (Sp->SphP[i].Density * All.cf_a3inv)) * All.MaxSfrTimescale;

  factorEVP = pow(Sp->SphP[i].Density * All.cf_a3inv / All.PhysDensThresh, -0.8) * All.FactorEVP;

  egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

  ne = Sp->SphP[i].Ne;
  tcool = GetCoolingTime(egyhot, Sp->SphP[i].Density * All.cf_a3inv, &ne, &gs, &DoCool);

  // Gadget-3 style calculation of cold cloud fraction
  y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
  
  // Careful handling to avoid numerical issues
  if(y <= 0) {
    x = 0;  // No cold clouds
  } else {
    x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));
    if(x < 0) x = 0;  // Numerical safety
    if(x > 1) x = 1;  // Cap at 100%
  }

  cloudmass = x * Sp->P[i].getMass();
  *xcloud = x;
  
  rateOfSF = (1 - All.FactorSN) * cloudmass / tsfr;

  /* convert to solar masses per yr */
  rateOfSF *= (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

  return rateOfSF;
}

/**
 * Create a star particle from a gas particle with additional debugging
 */
void coolsfr::create_star_particle(simparticles *Sp, int i, double prob, double rnd, double currentTemp)
{
  if(rnd < prob)  /* Make a star */
    {
      // Enhanced debug output
      if(ThisTask == 0 && All.StarFormationDebugLevel) {
        double rho = Sp->SphP[i].Density * All.cf_a3inv;
        double redshift = 1.0/All.Time - 1.0;
        printf("[SF] Creating star from particle %d at rho=%g, T=%g K, z=%g (prob=%g, rand=%g)\n", 
               Sp->P[i].ID.get(), rho, currentTemp, redshift, prob, rnd);
      }
      
      // CONVERT GAS INTO A STAR
      if(Sp->P[i].getMass() < 1.5 * All.TargetGasMass / GENERATIONS)
        {
          // Convert the gas particle into a star
          Sp->P[i].setType(4);  // Change type to star
          
          // The scalar Metallicity should already be set correctly from the gas
          // If it's not, ensure it's copied from Metals[0]
          #ifdef COOLING
          Sp->P[i].Metallicity = Sp->SphP[i].Metals[0];
          #endif
          
          #ifdef STELLARAGE
          // Record stellar formation time
          Sp->P[i].StellarAge = All.Time;
          #endif

          // Initialize feedback flag to 0 (no feedback applied yet)
          Sp->P[i].FeedbackFlag = 0;
        }
      // SPAWN A NEW STAR PARTICLE
      else
        {
          double pmass = Sp->P[i].getMass();
      
          // Make sure we have enough memory
          if(Sp->NumPart + 1 > Sp->MaxPart)
            Terminate("COOLSFR: No space left for star particle creation");
      
          // Create new star particle in the last position
          int j = Sp->NumPart;
      
          Sp->P[j] = Sp->P[i];  // Copy properties from parent
          Sp->P[j].setType(4);  // Set type to star
      
          // Transfer metallicity from gas to star (in case it wasn't copied by Sp->P[j] = Sp->P[i])
          #ifdef COOLING
          Sp->P[j].Metallicity = Sp->SphP[i].Metals[0];
          #endif
          
          // Adjust masses
          Sp->P[j].setMass(All.TargetGasMass / GENERATIONS);
          Sp->P[i].setMass(pmass - Sp->P[j].getMass());
      
          #ifdef STELLARAGE
          // Record stellar formation time
          Sp->P[j].StellarAge = All.Time;
          #endif

          // Initialize feedback flag to 0 (no feedback applied yet)
          Sp->P[j].FeedbackFlag = 0;
          
          // Increment particle counter
          Sp->NumPart++;
        }

        STARFORMATION_PRINT("New star is born! id=%d, probability=%g, rand=%g, temp=%g, density=%g, PhysDensThresh=%g\n", 
          Sp->P[i].ID.get(), prob, rnd, currentTemp, Sp->SphP[i].Density * All.cf_a3inv, All.PhysDensThresh);
    }
}



/**
 * Improved cooling_and_starformation function that more closely replicates Gadget-3 behavior
 * This eliminates the artificial vertical density cutoff in phase diagrams
 * Added additional temperature check to prevent hot gas from forming stars
 */
void coolsfr::cooling_and_starformation(simparticles *Sp)
{
  TIMER_START(CPU_COOLING_SFR);
  
  All.set_cosmo_factors_for_current_time();
  
  double sum_sm = 0, total_sm = 0, rate = 0, sum_mass_stars = 0, total_sum_mass_stars = 0;
  double rate_in_msunperyear = 0, totsfrrate = 0;
  double w = 0; // Random number for metallicity update
  double cum_mass_stars = 0; // Cumulative mass of stars formed

  double time_h_a = (All.ComovingIntegrationOn) ? All.Time * All.cf_hubble_a : 1.0;
  double total_sfr = 0;  // Total star formation rate
  int sf_eligible = 0;

  gas_state gs = GasState;
  do_cool_data DoCool = DoCoolData;
  
  // Debug output the G3-like parameter values we're using
  if(ThisTask == 0 && All.StarFormationDebugLevel) {
    mpi_printf("STARFORMATION: Using G3-like params: CritPhysDensity=%g, MaxSfrTimescale=%g, FactorSN=%g, FactorEVP=%g\n", 
              All.CritPhysDensity, All.MaxSfrTimescale, All.FactorSN, All.FactorEVP);
    mpi_printf("STARFORMATION: TempSupernova=%g, TempClouds=%g, CritOverDensity=%g, MaxStarFormationTemp=%g\n",
              All.TempSupernova, All.TempClouds, All.CritOverDensity, All.MaxStarFormationTemp);
  }
  
  // Calculate current redshift
  double redshift = 1.0/All.Time - 1.0;
  
  // ---------- FIRST LOOP: EVALUATE STAR FORMATION ELIGIBILITY ----------
  for(int i = 0; i < Sp->TimeBinsHydro.NActiveParticles; i++)
    {
      int target = Sp->TimeBinsHydro.ActiveParticleList[i];
      if(Sp->P[target].getType() == 0)  // Gas particle
        {
          if(Sp->P[target].getMass() == 0 && Sp->P[target].ID.get() == 0)
            continue;  // Skip particles that have been swallowed or eliminated
          
          // Get timestep
          double dt = (Sp->P[target].getTimeBinHydro() ? (((integertime)1) << Sp->P[target].getTimeBinHydro()) : 0) * All.Timebase_interval;
          double dtime = All.cf_atime * dt / All.cf_atime_hubble_a;
          
          // Check density criterion for star formation
          Sp->SphP[target].SfFlag = 0;  // Default: no star formation
          
          double utherm = Sp->get_utherm_from_entropy(target);
          double rho = Sp->SphP[target].Density * All.cf_a3inv;
          double currentTemp = convert_u_to_temp(utherm, rho, &Sp->SphP[target].Ne, &gs, &DoCool);
          
          // Strict temperature check to ensure only cold gas can form stars
          if(currentTemp > All.MaxStarFormationTemp) {
            // Skip particles that are too hot
            if(ThisTask == 0 && All.StarFormationDebugLevel > 2) {
              printf("[SF_DEBUG] Particle %d too hot: T=%g K > %g K\n", 
                     Sp->P[target].ID.get(), currentTemp, All.MaxStarFormationTemp);
            }
            continue;
          }
          
          // ---------- GADGET-3 STYLE CRITERIA WITH IMPROVEMENTS ----------
          
          // Base physical density threshold (same as G3)
          double base_threshold = All.PhysDensThresh;
          
          // 1. Redshift-dependent modifier (stronger effect than before)
          double z_factor = 1.0;
          if(redshift < 3.0) {
            // More aggressive reduction at low redshift - linearly decreases from z=3 to z=0
            z_factor = 0.5 + 0.5 * (redshift / 3.0);
            
            // Ensure we don't go below 40% of the original threshold
            if(z_factor < 0.4) z_factor = 0.4;
          }
          
          // 2. Temperature-dependent modifier (creates slope in phase space)
          double temp_ratio = currentTemp / All.MaxStarFormationTemp;
          double temp_factor;
          
          if(temp_ratio <= 0.2) {
            // For very cold gas, no modification
            temp_factor = 1.0;
          }
          else if(temp_ratio >= 0.9) {
            // Very warm gas - high threshold (effectively blocks star formation)
            temp_factor = 10.0;
          }
          else {
            // Smooth transition - quadratic increase with temperature
            double normalized = (temp_ratio - 0.2) / 0.7;  // 0 to 1 as temp goes from 20% to 90%
            temp_factor = 1.0 + 9.0 * normalized * normalized;  // Quadratic rise from 1 to 10
          }
          
          // 3. Add randomness to break the perfect boundary
          // Mix particle properties into the randomness source
          unsigned int seed = Sp->P[target].ID.get() + 
                             (unsigned int)(1000.0 * Sp->SphP[target].Entropy) + 
                             (unsigned int)(31415.0 * rho / All.PhysDensThresh);
          double random_factor = 0.85 + 0.3 * get_random_number(seed);
          
          // 4. Combine all factors (with randomness) to get effective threshold
          double effective_threshold = base_threshold * z_factor * temp_factor * random_factor;
          
          // 5. Create a density-dependent temperature threshold (crucial for diagonal boundary)
          // As density increases above threshold, allowed temperature increases
          double max_temp_threshold = All.MaxStarFormationTemp;
          if(rho > base_threshold) {
            // Log-slope in density space creates diagonal line in log-log phase space
            double excess_density_factor = log10(rho / base_threshold);
            
            // Scale allowed temperature with density excess
            // Tuned coefficient (0.3) controls the slope of the diagonal boundary
            max_temp_threshold *= (1.0 + 0.3 * excess_density_factor);
          }
          
          // Store the calculated temperature threshold for later use
          //Sp->SphP[target].MaxAllowedTemp = max_temp_threshold;
          
          // Apply the combined criteria for SF eligibility
          if(currentTemp < max_temp_threshold && rho >= effective_threshold) {
            Sp->SphP[target].SfFlag = 1;
            sf_eligible++;
            
            // Debug output if enabled
            if(ThisTask == 0 && All.StarFormationDebugLevel > 1) {
              printf("[SF_DEBUG] Particle %d eligible: rho=%g, T=%g, rho_thresh=%g, T_max=%g (z=%g z_f=%g, t_f=%g, r_f=%g)\n",
                    Sp->P[target].ID.get(), rho, currentTemp, effective_threshold, max_temp_threshold,
                    redshift, z_factor, temp_factor, random_factor);
            }
          }
          
          // Additional cosmological threshold check (same as original G3)
          if(All.ComovingIntegrationOn && Sp->SphP[target].Density < All.OverDensThresh)
            Sp->SphP[target].SfFlag = 0;

#ifdef WINDS
          // Handle wind particles
          if(Sp->SphP[target].DelayTime > 0)
            {
              Sp->SphP[target].SfFlag = 0;  // No star formation for particles in the wind
              
              // Decrement delay time for wind particles
              Sp->SphP[target].DelayTime -= dtime;
              
              // Check if wind particle has escaped to low density region
              if(Sp->SphP[target].DelayTime > 0)
                if(rho < All.WindFreeTravelDensFac * All.PhysDensThresh)
                  Sp->SphP[target].DelayTime = 0;
              
              // Ensure delay time doesn't go negative
              if(Sp->SphP[target].DelayTime < 0)
                Sp->SphP[target].DelayTime = 0;
            }
#endif
        } // End Type == 0
    } // End first loop
  
  if(ThisTask == 0)
      mpi_printf("STARFORMATION: %d particles eligible for star formation\n", sf_eligible);
  
  // ---------- SECOND LOOP: APPLY COOLING AND STAR FORMATION ----------
  for(int i = 0; i < Sp->TimeBinsHydro.NActiveParticles; i++)
    {
      int target = Sp->TimeBinsHydro.ActiveParticleList[i];
      if(Sp->P[target].getType() == 0)  // Gas particle
        {
          // Get timestep
          double dt = (Sp->P[target].getTimeBinHydro() ? (((integertime)1) << Sp->P[target].getTimeBinHydro()) : 0) * All.Timebase_interval;
          double dtime = All.cf_atime * dt / All.cf_atime_hubble_a;
          
          // Process star-forming particles
          if(Sp->SphP[target].SfFlag == 1)
            {
              // Calculate cloud mass fraction
              double xcloud;
              double sfr = get_starformation_rate(target, &xcloud, Sp);
              Sp->SphP[target].Sfr = sfr;
              
              // SfFlag might have been reset in get_starformation_rate due to temperature
              if(Sp->SphP[target].SfFlag == 0) {
                // Star formation has been aborted - do cooling instead
                cool_sph_particle(Sp, target, &gs, &DoCool);
                continue;
              }
              
              total_sfr += sfr;

              // Calculate star formation probability
              double tsfr, factorEVP, egyhot, egyeff, egyold, egycurrent;
              double ne = Sp->SphP[target].Ne;
              double rho = Sp->SphP[target].Density * All.cf_a3inv;
              
              // Standard model from Gadget-3
              tsfr = sqrt(All.PhysDensThresh / rho) * All.MaxSfrTimescale;
              factorEVP = pow(rho / All.PhysDensThresh, -0.8) * All.FactorEVP;
              egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

              double cloudmass = xcloud * Sp->P[target].getMass();
              if(tsfr < dtime) tsfr = dtime;
              
              // Probability of star formation - similar to Gadget-3 approach
              double p = (1 - All.FactorSN) * dtime / tsfr * cloudmass / Sp->P[target].getMass();
              double prob;
              
              if(Sp->P[target].getMass() < 1.5 * All.TargetGasMass / GENERATIONS)
                prob = 1 - exp(-p);
              else
                prob = Sp->P[target].getMass() / (All.TargetGasMass / GENERATIONS) * (1 - exp(-p));
              
              // Apply effective equation of state for star-forming gas - like Gadget-3
              if(dt > 0)
                {
                  // Calculate energy from entropy
                  egycurrent = Sp->SphP[target].Entropy * pow(Sp->SphP[target].Density, GAMMA_MINUS1) / GAMMA_MINUS1;
                  
                  // Calculate relaxation timescale - similar to Gadget-3
                  double trelax = tsfr * (1 - xcloud) / xcloud / (All.FactorSN * (1 + factorEVP));
                  double relaxfactor = exp(-dtime / trelax);
                  
                  // Update energy with relaxation
                  egyeff = egyhot * (1 - xcloud) + All.EgySpecCold * xcloud;
                  
                  // Apply equation of state - update entropy
                  Sp->SphP[target].Entropy = (egyeff + (egycurrent - egyeff) * relaxfactor) * GAMMA_MINUS1 /
                    pow(Sp->SphP[target].Density, GAMMA_MINUS1);
                  
                  // Reset entropy change rate
                  Sp->SphP[target].DtEntropy = 0;
                  
                  // Update thermodynamics
                  Sp->SphP[target].set_thermodynamic_variables();
                }
              
              // Random number for star formation decision
              double rnd = get_random_number(Sp->P[target].ID.get() + target + All.NumCurrentTiStep);
              
              // Perform final temperature check before creating star
              double currentTemp = convert_u_to_temp(Sp->get_utherm_from_entropy(target), 
                                  Sp->SphP[target].Density * All.cf_a3inv, &Sp->SphP[target].Ne, &gs, &DoCool);
                                  
              // Re-check temperature threshold before forming star
              if(currentTemp > All.MaxStarFormationTemp) {
                // Too hot to form stars - skip star formation
                if(ThisTask == 0 && All.StarFormationDebugLevel) {
                  printf("[SF_WARNING] Skipping too hot particle %d: T=%g K > %g K threshold\n", 
                        Sp->P[target].ID.get(), currentTemp, All.MaxStarFormationTemp);
                }
                // Set star formation rate to 0
                Sp->SphP[target].Sfr = 0;
                // Do cooling instead
                cool_sph_particle(Sp, target, &gs, &DoCool);
              }
              else {
                // Create star if probability check passes
                create_star_particle(Sp, target, prob, rnd, currentTemp);
                
#ifdef WINDS
                // Handle wind model if particle hasn't been turned into a star
                if(Sp->P[target].getType() == 0)
                  spawn_wind_particle(Sp, target, cloudmass, tsfr);
#endif
              }
            }
          else
            {
              // Normal cooling for non-star-forming particles
              Sp->SphP[target].Sfr = 0;
              
              // Perform standard cooling
              cool_sph_particle(Sp, target, &gs, &DoCool);
            }
        }
    }

  log_sfr(Sp);

  TIMER_STOP(CPU_COOLING_SFR);
}

/**
 * Function to get a random number for a given seed
 * Returns a random number between 0 and 1
 */
 double coolsfr::get_random_number(int id)
 {
   // Mix the seed with current time to increase randomness
   unsigned int seed = (unsigned int)id;
   seed = seed ^ (unsigned int)(All.Time * 1000000.0);
   
   // Add a mixing step (simple hash function)
   seed = (seed * 2654435761U) % UINT_MAX;
   
   // Use the standard random number generator steps
   int ia = 16807;
   int im = 2147483647;
   int iq = 127773;
   int ir = 2836;
   
   int k = seed / iq;
   unsigned int temp = ia * (seed - k * iq) - ir * k;
   if(temp < 0) temp += im;
   
   double result = temp / (double)im;
   
   // For debugging
   if(ThisTask == 0)
   {
     static int rnd_checks = 0;
     if(rnd_checks < 5)
     {
       printf("STARFORMATION: random id=%d, seed=%u, result=%g\n", id, seed, result);
       rnd_checks++;
     }
   }
   
   return result;
 }

/**
 * Create a wind particle from a gas particle
 */
void coolsfr::spawn_wind_particle(simparticles *Sp, int i, double cloudmass, double tsfr)
{
#ifdef WINDS
  if(All.WindEfficiency <= 0)
    return;
  
  double p = All.WindEfficiency * (1 - All.FactorSN) * cloudmass / tsfr * (All.UnitTime_in_s / (All.cf_atime_hubble_a)) / Sp->P[i].getMass();
  
  if(p < 0)
    Terminate("COOLSFR: Negative wind probability p=%g", p);
  
  if(get_random_number(Sp->P[i].ID.get() + 2) < p)
    {
      // Calculate wind velocity
      double v = sqrt(2 * All.WindEnergyFraction * All.FactorSN * All.EgySpecSN / (1 - All.FactorSN) / All.WindEfficiency);
      
      // Save particle velocity
      double vx = Sp->P[i].Vel[0];
      double vy = Sp->P[i].Vel[1];
      double vz = Sp->P[i].Vel[2];
      
      // Get random direction
      double theta = acos(2 * get_random_number(Sp->P[i].ID.get() + 3) - 1);
      double phi = 2 * M_PI * get_random_number(Sp->P[i].ID.get() + 4);
      
      // Add kick in random direction
      Sp->P[i].Vel[0] += v * sin(theta) * cos(phi);
      Sp->P[i].Vel[1] += v * sin(theta) * sin(phi);
      Sp->P[i].Vel[2] += v * cos(theta);
      
      // Update delay time
      Sp->SphP[i].DelayTime = All.WindFreeTravelLength / (v / All.cf_atime);
      
#ifdef NOWINDTIMECONSTRAINT
      // Remove maximum time constraint if enabled
      Sp->SphP[i].DelayTime = FLT_MAX;
#endif
    }
#endif
}

/**
 * Integrate the star formation rate to validate the model
 */
void coolsfr::integrate_sfr(void)
{
  double rho0, rho, rho2, q, dz, gam, sigma = 0, sigma_u4, sigmasfr = 0, ne, P1;
  double x = 0, y, P, P2, x2, y2, tsfr2, factorEVP2, egyhot2, tcool2, drho, dq;
  double meanweight, u4, z, tsfr, tcool, egyhot, factorEVP, egyeff, egyeff2;
  FILE *fd;
  gas_state gs = GasState;
  do_cool_data DoCool = DoCoolData;

  meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));	/* assuming FULL ionization */
  u4 = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * 1.0e4;
  u4 *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

  // Save time state
  double oldtime = All.Time;
  
  if(All.ComovingIntegrationOn)
    {
      All.Time = 1.0;		/* to be guaranteed to get z=0 rate */
      IonizeParams();
    }

  if(ThisTask == 0)
    fd = fopen("eos.txt", "w");
  else
    fd = 0;

  // Generate table of effective pressure vs density
  for(rho = All.PhysDensThresh; rho <= 1000 * All.PhysDensThresh; rho *= 1.1)
    {
      tsfr = sqrt(All.PhysDensThresh / rho) * All.MaxSfrTimescale;

      factorEVP = pow(rho / All.PhysDensThresh, -0.8) * All.FactorEVP;

      egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

      ne = 1.0;
#ifdef TABLECOOL
#ifdef METALCOOL
      tcool = GetCoolingTime(egyhot, rho, &ne, &gs, &DoCool);
#else
      tcool = GetCoolingTime(egyhot, rho, &ne, &gs, &DoCool);
#endif
#else
      tcool = GetCoolingTime(egyhot, rho, &ne, &gs, &DoCool);
#endif

      y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
      x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));

      egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

      P = GAMMA_MINUS1 * rho * egyeff;

      if(ThisTask == 0 && fd)
        {
          fprintf(fd, "%g %g\n", rho, P);
        }
    }

  if(ThisTask == 0 && fd)
    fclose(fd);

  if(ThisTask == 0)
    fd = fopen("sfrrate.txt", "w");
  else
    fd = 0;

  // Calculate star formation rate vs density in the disk model
  for(rho0 = All.PhysDensThresh; rho0 <= 10000 * All.PhysDensThresh; rho0 *= 1.02)
    {
      z = 0;
      rho = rho0;
      q = 0;
      dz = 0.001;

      sigma = sigmasfr = sigma_u4 = 0;

      while(rho > 0.0001 * rho0)
        {
          if(rho > All.PhysDensThresh)
            {
              tsfr = sqrt(All.PhysDensThresh / rho) * All.MaxSfrTimescale;

              factorEVP = pow(rho / All.PhysDensThresh, -0.8) * All.FactorEVP;

              egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

              ne = 1.0;

#ifdef TABLECOOL
#ifdef METALCOOL
              tcool = GetCoolingTime(egyhot, rho, &ne, &gs, &DoCool);
#else
              tcool = GetCoolingTime(egyhot, rho, &ne, &gs, &DoCool);
#endif
#else
              tcool = GetCoolingTime(egyhot, rho, &ne, &gs, &DoCool);
#endif

              y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
              x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));

              egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

              P = P1 = GAMMA_MINUS1 * rho * egyeff;

              rho2 = 1.1 * rho;
              tsfr2 = sqrt(All.PhysDensThresh / rho2) * All.MaxSfrTimescale;
              factorEVP2 = pow(rho2 / All.PhysDensThresh, -0.8) * All.FactorEVP;
              egyhot2 = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;
#ifdef TABLECOOL
#ifdef METALCOOL
              tcool2 = GetCoolingTime(egyhot2, rho2, &ne, &gs, &DoCool);
#else
              tcool2 = GetCoolingTime(egyhot2, rho2, &ne, &gs, &DoCool);
#endif
#else
              tcool2 = GetCoolingTime(egyhot2, rho2, &ne, &gs, &DoCool);
#endif
              y2 = tsfr2 / tcool2 * egyhot2 / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
              x2 = 1 + 1 / (2 * y2) - sqrt(1 / y2 + 1 / (4 * y2 * y2));
              egyeff2 = egyhot2 * (1 - x2) + All.EgySpecCold * x2;
              P2 = GAMMA_MINUS1 * rho2 * egyeff2;

              gam = log(P2 / P1) / log(rho2 / rho);
            }
          else
            {
              tsfr = 0;

              P = GAMMA_MINUS1 * rho * u4;
              gam = 1.0;

              sigma_u4 += rho * dz;
            }

          drho = q;
          dq = -(gam - 2) / rho * q * q - 4 * M_PI * All.G / (gam * P) * rho * rho * rho;

          sigma += rho * dz;
          if(tsfr > 0)
            {
              sigmasfr += (1 - All.FactorSN) * rho * x / tsfr * dz;
            }

          rho += drho * dz;
          q += dq * dz;
        }

      sigma *= 2;		/* to include the other side */
      sigmasfr *= 2;
      sigma_u4 *= 2;

      if(ThisTask == 0 && fd)
        {
          fprintf(fd, "%g %g %g %g\n", rho0, sigma, sigmasfr, sigma_u4);
        }
    }

  // Restore time state
  if(All.ComovingIntegrationOn)
    {
      All.Time = oldtime;
      IonizeParams();
    }

  if(ThisTask == 0 && fd)
    fclose(fd);
}

/**
 * Rearrange particles after star formation to maintain proper ordering
 */
void coolsfr::rearrange_particle_sequence(simparticles *Sp)
{
  /* This function ensures that particles are ordered properly after star formation
   * - Gas particles must come first (Type 0)
   * - Then stars and other particle types
   */
  
  int swap_count = 0;
  
  // First, identify the last gas particle position
  int last_gas = Sp->NumGas - 1;
  
  // Find non-gas particles in the gas section
  for(int i = 0; i < Sp->NumGas; i++)
    {
      if(Sp->P[i].getType() != 0)  // Found a non-gas particle in gas section
        {
          // Find a gas particle in the non-gas section to swap with
          while(last_gas >= 0 && Sp->P[last_gas].getType() != 0)
            last_gas--;
          
          if(last_gas < 0 || last_gas <= i)
            break;  // No more swapping needed
          
          // Swap the particles
          sph_particle_data sph_tmp = Sp->SphP[i];
          Sp->SphP[i] = Sp->SphP[last_gas];
          Sp->SphP[last_gas] = sph_tmp;
          
          particle_data p_tmp = Sp->P[i];
          Sp->P[i] = Sp->P[last_gas];
          Sp->P[last_gas] = p_tmp;
          
          swap_count++;
          last_gas--;
        }
    }
  
  // Update NumGas - count gas particles
  int new_numgas = 0;
  for(int i = 0; i < Sp->NumPart; i++)
    {
      if(Sp->P[i].getType() == 0)
        new_numgas++;
    }
  
  // Update global counters
  int diff = Sp->NumGas - new_numgas;
  Sp->NumGas = new_numgas;
  
  int tot_diff;
  MPI_Allreduce(&diff, &tot_diff, 1, MPI_INT, MPI_SUM, Communicator);
  All.TotN_gas -= tot_diff;
  
  if(ThisTask == 0 && swap_count > 0)
    printf("COOLSFR: Rearranged %d particles after star formation.\n", swap_count);
}

/**
 * Write star formation rate to sfr.txt
 * This is a simple version that matches what's working in starformation.cc
 */
void coolsfr::log_sfr(simparticles *Sp)
{
    // Column 1: Time (scale factor)
    double time = All.Time;
    
    // Column 2: Redshift
    double z = 1.0 / All.Time - 1.0;
    
    // Calculate SFR and related quantities
    double total_sfr = 0.0;        // Total SFR in code units
    double total_gas_mass = 0.0;   // Total gas mass
    double total_sfr_mass = 0.0;   // Total mass in star-forming gas
    int sfr_count = 0;             // Number of star-forming gas particles
    
    // Loop through all gas particles
    for(int i = 0; i < Sp->NumGas; i++)
    {
      // Add to total gas mass
      total_gas_mass += Sp->P[i].getMass();
      
      // Check if this particle has SFR > 0
      if(Sp->SphP[i].Sfr > 0)
      {
        total_sfr += Sp->SphP[i].Sfr;
        total_sfr_mass += Sp->P[i].getMass();
        sfr_count++;
      }
    }
    
    // Calculate derived quantities
    double box_size_mpc = All.BoxSize / 1000.0;  // Convert kpc to Mpc
    double volume = pow(box_size_mpc, 3.0);      // Volume in Mpc^3
    
    // Convert to physical units
    // SFR from code units to Msun/yr
    double sfr_in_msun_per_year = total_sfr * (All.UnitMass_in_g / SOLAR_MASS) / 
                                  (All.UnitTime_in_s / SEC_PER_YEAR) / All.HubbleParam;
    
    // Calculate SFR density
    double sfr_density = sfr_in_msun_per_year / volume;
    
    // Convert masses to Msun
    double gas_mass_msun = total_gas_mass * (All.UnitMass_in_g / SOLAR_MASS) / All.HubbleParam;
    double sfr_mass_msun = total_sfr_mass * (All.UnitMass_in_g / SOLAR_MASS) / All.HubbleParam;
    
    // For multi-process runs, collect data from all processes
    double global_sfr, global_gas_mass;
    int global_sfr_count;

    MPI_Reduce(&total_sfr, &global_sfr, 1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
    MPI_Reduce(&total_gas_mass, &global_gas_mass, 1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
    MPI_Reduce(&sfr_count, &global_sfr_count, 1, MPI_INT, MPI_SUM, 0, Communicator);
    
    // Only write from the root task
    if(ThisTask == 0 && Logs.FdSfr)
    {
      // Write to SFR file
      fprintf(Logs.FdSfr, "%14e %14e %14e %14e %14e %d\n",
              time,               // Column 1: Scale factor
              z,                  // Column 2: Redshift
              global_sfr,          // Column 3: SFR [Msun/yr]
              sfr_density,        // Column 4: SFR density [Msun/yr/Mpc^3]
              global_gas_mass,      // Column 5: Gas mass [Msun]
              global_sfr_count);         // Column 6: Number of star-forming gas cells
      
      fflush(Logs.FdSfr);
      
      // Debug output
      printf("[SFR_LOG] z=%g SF particles=%d SFR=%g Msun/yr SFRD=%g Msun/yr/Mpc^3\n", 
             z, sfr_count, sfr_in_msun_per_year, sfr_density);
    }
}

#endif /* STARFORMATION */
#endif /* COOLING */