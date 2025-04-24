/*******************************************************************************
 Uses a polytropic equation of state with variable effective pressure and 
 thermodynamics that mimic a multiphase ISM.

 main functions:
 --------------------------
 sf_evaluate_particle()       : Determines if a gas particle is eligible for star formation
 get_starformation_rate()     : Calculates the SFR for a gas particle
 update_thermodynamic_state() : Updates the thermodynamic state of star-forming gas
 cooling_and_starformation()  : Main routine that handles cooling and star formation
 winds_effective_model()      : Implements the wind feedback model
 *******************************************************************************/

/*! \file sfr_eff.cc
 *
 *  \brief Module for the effective multi-phase model of star formation and feedback
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
 #include <algorithm>
 
 #include "../cooling_sfr/cooling.h"
 #include "../data/allvars.h"
 #include "../data/dtypes.h"
 #include "../data/mymalloc.h"
 #include "../logs/logs.h"
 #include "../logs/timer.h"
 #include "../system/system.h"
 #include "../time_integration/timestep.h"
 
  // Debug output function
  #define SF_PRINT(...) \
    do { if (All.StarFormationDebug) mpi_printf(__VA_ARGS__); } while (0)

 /**
  * The following routines implement the star formation and feedback model
  * from Springel & Hernquist 2003, MNRAS, 339, 289 (Multi-phase model).
  * The approach is based on a subgrid model where star formation and
  * feedback are treated with effective equations derived from a model for
  * the unresolved ISM structure.
  */
 
 /** \brief Get metallicity for a particle
  * 
  * This function calculates and returns the metallicity of a particle
  * normalized to solar units.
  * 
  * \param i Index of the particle
  * \return Total metallicity (normalized to solar units)
  */
 double coolsfr::getZ(simparticles *Sp, int i)
 {
 #ifdef METALS
   return Sp->P[i].Metallicity / SOLAR_METALLICITY;
 #else
   return 0.0;
 #endif
 }
 
 /**
  * \brief Calculate threshold density for star formation based on physical conditions
  * 
  * This function computes the physical density threshold for star formation
  * based on the Jeans length criterion and other physical considerations.
  */
 void coolsfr::init_clouds(void)
 {
   // Set default hydrogen mass fraction
   GasState.XH = HYDROGEN_MASSFRAC;
 
   double A0, dens, tcool, ne, coolrate, egyhot, x, u4, meanweight;
   double thresholdStarburst;
 
   if(All.PhysDensThresh == 0)
     {
       A0 = All.FactorEVP;
       egyhot = All.EgySpecSN / A0;
 
       // Calculate energy at 10^4 K
       meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));  // Assuming FULL ionization
       //u4 = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * 1.0e4;   // Gadget-3 version
       u4 = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * 7500;    // make it easier to hit cooling thresholds
       u4 *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;
 
       // Set reference density
       if(All.ComovingIntegrationOn)
         dens = 1.0e6 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G);
       else
         // Physical cgs value, converted to system units
         dens = 9.205e-24 / (All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam);
 
       // Set up for cooling calculation
       do_cool_data DoCool{}; 
       gas_state gs = GasState;
       All.Time = 1.0;  // Ensure z=0 rate
       IonizeParams();
 
       ne = 1.0;
       SetZeroIonization();
       
       // Calculate cooling time
       tcool = GetCoolingTime(egyhot, dens, &ne, &gs, &DoCool);
 
       coolrate = egyhot / tcool / dens;
 
       // Calculate expected cold gas fraction at threshold
       x = (egyhot - u4) / (egyhot - All.EgySpecCold);
 
       // Compute physical density threshold
       All.PhysDensThresh =
         x / pow(1 - x, 2) * (All.FactorSN * All.EgySpecSN - (1 -
                               All.FactorSN) * All.EgySpecCold) /
         (All.MaxSfrTimescale * coolrate);
 
       // Report calculations
       mpi_printf("STARFORMATION: A0= %g\n", A0);
       mpi_printf("STARFORMATION: Computed PhysDensThresh= %g (internal units) %g h^2 cm^-3\n", All.PhysDensThresh,
              All.PhysDensThresh / (PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs));
       mpi_printf("STARFORMATION: Expected cold gas fraction at threshold = %g\n", x);
       mpi_printf("STARFORMATION: tcool=%g dens=%g egyhot=%g\n", tcool, dens, egyhot);
 
 #ifdef H2_SF_MODEL
       // Use fixed threshold for H2-regulated SF
       All.PhysDensThresh = (DENSTHRESH/(All.HubbleParam*All.HubbleParam)) * 
                            PROTONMASS / (HYDROGEN_MASSFRAC*All.UnitDensity_in_cgs);
       
       // Calculate surface density normalization
       double pctocm = 3.085678e18;
       double gtomsun = 1.989e33;
       All.SIGMA_NORM = 1./((All.UnitMass_in_g/(All.UnitLength_in_cm*All.UnitLength_in_cm))*
                            ((pctocm*pctocm)/gtomsun));
       
       mpi_printf("STARFORMATION: Using H2-regulated star formation model\n");
       mpi_printf("STARFORMATION: Setting PhysDensThresh=%g in H2 model (system unit) %g h^2 cm^-3\n", 
                  All.PhysDensThresh,
                  All.PhysDensThresh / (PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs));
 #endif
 
       // Calculate where runaway SF sets in
       dens = All.PhysDensThresh * 10;
       double tsfr, factorEVP, egyeff, neff, fac, peff;
       double y;
 
       do
         {
           // Calculate SF timescale and evaporation factor
           tsfr = sqrt(All.PhysDensThresh / (dens)) * All.MaxSfrTimescale;
           factorEVP = pow(dens / All.PhysDensThresh, -0.8) * All.FactorEVP;
           egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;
 
           // Calculate cooling time
           ne = 0.5;
           tcool = GetCoolingTime(egyhot, dens, &ne, &gs, &DoCool);
 
           // Calculate cloud mass fraction and effective energy
           y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
           x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));
           egyeff = egyhot * (1 - x) + All.EgySpecCold * x;
 
           // Calculate effective pressure
           peff = GAMMA_MINUS1 * dens * egyeff;
 
           // Calculate polytropic index
           fac = 1 / (log(dens * 1.025) - log(dens));
           dens *= 1.025;
           neff = -log(peff) * fac;
 
           // Recalculate with new density
           tsfr = sqrt(All.PhysDensThresh / (dens)) * All.MaxSfrTimescale;
           factorEVP = pow(dens / All.PhysDensThresh, -0.8) * All.FactorEVP;
           egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;
 
           ne = 0.5;
           tcool = GetCoolingTime(egyhot, dens, &ne, &gs, &DoCool);
 
           y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
           x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));
           egyeff = egyhot * (1 - x) + All.EgySpecCold * x;
 
           peff = GAMMA_MINUS1 * dens * egyeff;
 
           neff += log(peff) * fac;
         }
       while(neff > 4.0 / 3);  // Stop when polytropic index exceeds 4/3
 
       thresholdStarburst = dens;
 
       // Report starburst threshold
       mpi_printf("STARFORMATION: Run-away sets in for dens=%g\n", thresholdStarburst);
       mpi_printf("STARFORMATION: Dynamic range for quiescent star formation= %g\n", thresholdStarburst / All.PhysDensThresh);
 
       // Reset time for cosmological simulations
       if(All.ComovingIntegrationOn)
         {
           All.Time = All.TimeBegin;
           IonizeParams();
         }
 
 #ifdef WINDS
       // Report wind speed if winds are enabled
       if(All.WindEfficiency > 0)
         mpi_printf("STARFORMATION: Windspeed: %g\n",
              sqrt(2 * All.WindEnergyFraction * All.FactorSN * All.EgySpecSN / (1 - All.FactorSN) /
                   All.WindEfficiency));
 #endif
     }
 }
 
 /**
  * \brief Set up unit conversions and parameters for star formation model
  * 
  * This function initializes physical constants and parameters used in the
  * star formation model, converting them to internal code units.
  */
 void coolsfr::set_units_sfr(void)
 {
   // Calculate overdensity threshold in code units
   All.OverDensThresh =
     All.CritOverDensity * All.OmegaBaryon * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G);
 
   // Calculate physical density threshold in code units
   All.PhysDensThresh = All.CritPhysDensity * PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs;
 
   // Calculate cold cloud energy (assuming neutral gas)
   double meanweight = 4 / (1 + 3 * HYDROGEN_MASSFRAC);
   All.EgySpecCold = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.TempClouds;
   All.EgySpecCold *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;
 
   // Calculate supernova energy (assuming fully ionized gas)
   meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));
   All.EgySpecSN = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.TempSupernova;
   All.EgySpecSN *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;
 
   // Report derived values
   mpi_printf("STARFORMATION: OverDensThresh= %g\n", All.OverDensThresh);
   mpi_printf("STARFORMATION: PhysDensThresh= %g (internal units)\n", All.PhysDensThresh);
   mpi_printf("STARFORMATION: EgySpecCold= %g\n", All.EgySpecCold);
   mpi_printf("STARFORMATION: EgySpecSN= %g\n", All.EgySpecSN);
 }
 
 /**
  * \brief Determine if a gas particle should form stars
  * 
  * Evaluate whether a gas particle meets the criteria for star formation
  * based on density, temperature, etc.
  * 
  * \param Sp Pointer to simulation particles
  * \param i Index of the gas particle
  * \return true if the particle should form stars, false otherwise
  */
 bool coolsfr::sf_evaluate_particle(simparticles *Sp, int i)
 {
   // Check particle type
   if(Sp->P[i].getType() != 0)
     return false;
 
   // Make sure particle has non-zero mass
   if(Sp->P[i].getMass() == 0)
     return false;
 
   double rho = Sp->SphP[i].Density * All.cf_a3inv;  // Physical density
   double temp, utherm;
 
   // Get thermal energy and temperature
   utherm = Sp->get_utherm_from_entropy(i);
   do_cool_data DoCool{}; 
   gas_state gs = GasState;
   double ne = Sp->SphP[i].Ne;
   temp = convert_u_to_temp(utherm, rho, &ne, &gs, &DoCool);
 
   // 1. Check for density threshold
   if(rho < All.PhysDensThresh)
     return false;
 
   // 2. Check for temperature threshold
   if(temp > All.MaxStarFormationTemp)
     return false;
 
   // 3. Check for overdensity threshold in cosmological simulations
   if(All.ComovingIntegrationOn && Sp->SphP[i].Density < All.OverDensThresh)
     return false;
 
 #ifdef WINDS
   // Don't form stars in wind particles
   if(Sp->SphP[i].DelayTime > 0)
     return false;
 #endif
 
   // All criteria passed, particle can form stars
   return true;
 }
 
 /**
  * \brief Calculate star formation rate for a gas particle
  * 
  * This function calculates the star formation rate for a gas particle
  * based on its physical properties and the effective multi-phase model.
  * 
  * \param Sp Pointer to simulation particles
  * \param i Index of the gas particle
  * \param cloudMassFraction Pointer to store cloud mass fraction
  * \return Star formation rate in Msun/year
  */
 double coolsfr::get_starformation_rate(simparticles *Sp, int i, double *cloudMassFraction)
 {
   // Default: no molecular clouds
   *cloudMassFraction = 0.0;
   
   // Check if particle is eligible for star formation
   if(!sf_evaluate_particle(Sp, i))
     return 0.0;
 
   double rho = Sp->SphP[i].Density * All.cf_a3inv;  // Physical density
   double tsfr, egyhot, tcool, factorEVP, y, x;
   double cloudmass;
   
   // Standard density threshold model
   tsfr = sqrt(All.PhysDensThresh / rho) * All.MaxSfrTimescale;
   factorEVP = pow(rho / All.PhysDensThresh, -0.8) * All.FactorEVP;
   egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;
 
   double ne = Sp->SphP[i].Ne;
   do_cool_data DoCool{}; 
   gas_state gs = GasState;
   tcool = GetCoolingTime(egyhot, rho, &ne, &gs, &DoCool);
 
   // Calculate cloud mass fraction
   if(tcool > 0)
     {
       y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
       x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));
     }
   else
     {
       x = 1.0;  // Fully in cold phase if no cooling time
     }
 
   cloudmass = x * Sp->P[i].getMass();
   *cloudMassFraction = x;
   
   // Calculate star formation rate
   double rateOfSF = cloudmass / tsfr;
 
   // Convert to solar masses per year
   rateOfSF *= (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);
 
   return rateOfSF;
 }
 
 /**
  * \brief Update thermodynamic state for star-forming particles
  * 
  * This function updates the thermodynamic state (entropy, internal energy)
  * of a gas particle undergoing star formation according to the effective
  * multi-phase model.
  * 
  * \param Sp Pointer to simulation particles
  * \param i Index of the gas particle
  * \param dt Physical time step
  * \param cloudMassFraction Fraction of gas mass in cold clouds
  */
 void coolsfr::update_thermodynamic_state(simparticles *Sp, int i, double dt, double cloudMassFraction)
 {
   double rho = Sp->SphP[i].Density * All.cf_a3inv;  // Physical density
   double x = cloudMassFraction;
   double tsfr = sqrt(All.PhysDensThresh / rho) * All.MaxSfrTimescale;
   double factorEVP = pow(rho / All.PhysDensThresh, -0.8) * All.FactorEVP;
   
   // Calculate relaxation timescale
   double trelax = tsfr * (1 - x) / x / (All.FactorSN * (1 + factorEVP));
   double egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;
   double egyeff = egyhot * (1 - x) + All.EgySpecCold * x;
   
   // Current thermal energy
   double egycurrent = Sp->get_utherm_from_entropy(i);
   
   // Calculate relaxation factor
   double relaxfactor;
   if(trelax > 0)
     relaxfactor = exp(-dt / trelax);
   else
     relaxfactor = 0.0;
   
   // Apply relaxation to energy
   double egynew = egyeff + (egycurrent - egyeff) * relaxfactor;
   
   // Convert energy to entropy
   Sp->set_entropy_from_utherm(egynew, i);
   
   // Update thermodynamic variables
   Sp->SphP[i].set_thermodynamic_variables();
 }
 
 /**
  * \brief Wind model implementation for the effective multi-phase model
  * 
  * This function implements the wind model described in Springel & Hernquist (2003).
  * When active, it gives gas particles in star-forming regions a probability to
  * receive a velocity kick and enter a wind phase, during which they do not
  * participate in star formation for a delay time.
  * 
  * \param Sp Pointer to simulation particles
  * \param i Index of the gas particle
  * \param dt Physical time step
  * \param sfr Star formation rate
  * \param cloudMassFraction Fraction of gas mass in cold clouds
  */
 #ifdef WINDS
 void coolsfr::winds_effective_model(simparticles *Sp, int i, double dt, double sfr, double cloudMassFraction)
 {
   if(All.WindEfficiency <= 0 || Sp->P[i].getType() != 0)
     return;
 
   double rho = Sp->SphP[i].Density * All.cf_a3inv;  // Physical density
   
   // Probability of launching a wind
   double p = All.WindEfficiency * Sp->P[i].getMass() / All.TargetGasMass * dt * sfr / cloudMassFraction;
   
   if(get_random_number() < p)
     {
       // Calculate wind velocity
       double windSpeed = sqrt(2 * All.WindEnergyFraction * All.FactorSN * All.EgySpecSN / (1 - All.FactorSN));
       
       // Get a random direction
       double theta = acos(2 * get_random_number() - 1);
       double phi = 2 * M_PI * get_random_number();
       
       // Add kick to velocity
       Sp->P[i].Vel[0] += windSpeed * sin(theta) * cos(phi);
       Sp->P[i].Vel[1] += windSpeed * sin(theta) * sin(phi);
       Sp->P[i].Vel[2] += windSpeed * cos(theta);
       
       // Set delay time - particle is decoupled from SF during this time
       double windDelay = All.WindFreeTravelLength / windSpeed * All.cf_atime;
       Sp->SphP[i].DelayTime = windDelay;
       
       SF_PRINT("WINDS: Particle %d received wind kick, v=%g, delay=%g\n", Sp->P[i].ID.get(), windSpeed, windDelay);
     }
 }
 #endif
 
 /**
  * \brief Primary cooling and star formation routine
  * 
  * This function handles both cooling of gas and the conversion of gas
  * to stars according to the effective multi-phase model.
  */
 void coolsfr::cooling_and_starformation(simparticles *Sp)
 {
   TIMER_START(CPU_COOLING_SFR);
 
   All.set_cosmo_factors_for_current_time();
 
   gas_state gs = GasState;
   do_cool_data DoCool = DoCoolData;
 
   // Initialize statistics
   double sum_sm = 0.0;          // Total mass expected to be converted to stars
   double sum_mass_stars = 0.0;  // Total mass actually converted to stars
   stars_spawned    = 0;
   stars_converted  = 0;
 
   /* Apply isochoric cooling to all active gas particles */
   for(int i = 0; i < Sp->TimeBinsHydro.NActiveParticles; i++)
     {
       int target = Sp->TimeBinsHydro.ActiveParticleList[i];
       if(Sp->P[target].getType() == 0)
         {
           if(Sp->P[target].getMass() == 0 && Sp->P[target].ID.get() == 0)
             continue; /* skip particles that have been swallowed or eliminated */
 
           double dt = (Sp->P[target].getTimeBinHydro() ? (((integertime)1) << Sp->P[target].getTimeBinHydro()) : 0) * All.Timebase_interval;
           double dtime = All.cf_atime * dt / All.cf_atime_hubble_a;
           
           // Check if particle eligible for star formation
           bool sf_active = sf_evaluate_particle(Sp, target);
           
           // Regular cooling for non-star-forming particles
           if(!sf_active)
             {
               cool_sph_particle(Sp, target, &gs, &DoCool);
               Sp->SphP[target].Sfr = 0;
             }
           else
             {
               // For star-forming particles:
               double cloudmassfrac;
               double sfr = get_starformation_rate(Sp, target, &cloudmassfrac);
               
               // Store star formation rate in particle data
               Sp->SphP[target].Sfr = sfr;
               
               // Update stellar mass in time bins for statistics
               Sp->TimeBinSfr[Sp->P[target].getTimeBinHydro()] += sfr;
               
               // Calculate expected mass of stars to be formed
               double sm = cloudmassfrac * Sp->P[target].getMass() * (1 - exp(-sfr * dtime / Sp->P[target].getMass()));
               sum_sm += sm;
               
               // Create stars stochastically
               if(All.StarformationMode > 0)
                 {
                   // Calculate probability of forming a star
                   double p = sm / Sp->P[target].getMass(); // original Gadget-3 version
                   //double p = 1.0 - exp(-sm / Sp->P[target].getMass());  // newer
                   double randomnum = get_random_number();

                   // Random draw to determine if star forms
                   // GO LOOK AT Shayou's GADGET3s random number generator
                   if(randomnum < p)
                     {
 
                             if (sm >= Sp->P[target].getMass()) {
                                 // convert the entire gas particle into a star
                                 SF_PRINT("STARFORMATION: Reached star formation probability (convert)! random=%.4e is less than prob=%.4e\n", randomnum, p);
                                 stars_converted++;
                                 sum_mass_stars += Sp->P[target].getMass();
                                 convert_sph_particle_into_star(Sp, target, All.Time);
                             } else {
                                 // spawn a brand new star of mass = sm
                                 if(Sp->NumPart + stars_spawned >= Sp->MaxPart) {
                                     if(ThisTask == 0)
                                     SF_PRINT("WARNING: no space to spawn star for gas particle %d (skipping)\n",
                                             Sp->P[target].ID.get());
                                     continue;  // skip this spawn, but keep processing the rest
                                 }
                                 SF_PRINT("STARFORMATION: Reached star formation probability (spawn)! random=%.4e is less than prob=%.4e\n", randomnum, p);
                                 int j = Sp->NumPart + stars_spawned;
                                 spawn_star_from_sph_particle(Sp, target, All.Time, j, sm);
                                 sum_mass_stars += sm;
                                 stars_spawned++;
                             }
                         
                             SF_PRINT("STARFORMATION: Particle %d forms star with probability %g at z=%.3e\n", Sp->P[target].ID.get(), p, 1.0/All.Time - 1.0);
                       
                       // Add to the converted stellar mass
                       sum_mass_stars += Sp->P[target].getMass();
                     }
                 }
               
               // Update thermodynamic state according to effective model
               update_thermodynamic_state(Sp, target, dtime, cloudmassfrac);
               
 #ifdef WINDS
               // Apply wind model
               mpi_printf("WINDS IS ON!\n");
               winds_effective_model(Sp, target, dtime, sfr, cloudmassfrac);
 #endif
               
               // Update metallicity due to star formation
               if(Sp->P[target].getType() == 0) // Ensure still a gas particle
                 {
                   double Z = Sp->SphP[target].Metallicity;
                   Z += All.MetalYield * cloudmassfrac * (1 - exp(-sfr * dtime / Sp->P[target].getMass()));
                   Sp->SphP[target].Metallicity = Z;
                   Sp->P[target].Metallicity = Z;
                 }
             }
         }
     }
   
   // Collect star formation statistics
   double totsfrrate = 0.0;
   for(int bin = 0; bin < TIMEBINS; bin++)
     if(Sp->TimeBinsHydro.TimeBinCount[bin])
       totsfrrate += Sp->TimeBinSfr[bin];
   
   double total_sm, total_mass_stars;
   MPI_Reduce(&sum_sm, &total_sm, 1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
   MPI_Reduce(&sum_mass_stars, &total_mass_stars, 1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
   
   if(ThisTask == 0)
     {
       double rate;
       if(All.TimeStep > 0)
         rate = total_sm / (All.TimeStep / All.cf_atime_hubble_a);
       else
         rate = 0;
       
       // Compute rates in physical units
       double rate_in_msunperyear = rate * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);
       
       // Accumulate cumulative star mass
       cum_mass_stars += total_mass_stars;
       
       // Log star formation information
       fprintf(Logs.FdSfr, "%14e %14e %14e %14e %14e %14e\n", 
               All.Time, total_sm, totsfrrate, rate_in_msunperyear, total_mass_stars, cum_mass_stars);
       myflush(Logs.FdSfr);
       
       SF_PRINT("STARFORMATION: z=%g  SFR=%g Msun/yr  total_sm=%g  total_new_stars=%g\n",
                  1.0/All.Time - 1.0, rate_in_msunperyear, total_sm, total_mass_stars);
     }
 
   if(stars_spawned > 0) {
     Sp->NumPart += stars_spawned;
     Sp->TotNumPart += stars_spawned;
   }
   if(stars_converted > 0) {
     Sp->TotNumGas -= stars_converted;
   }
   int star_count = 0;
   for(int i = 0; i < Sp->NumPart; i++)
     if(Sp->P[i].getType() == 4)
       star_count++;

  // Calculate local star count
  int local_star_count = 0;
  for(int i = 0; i < Sp->NumPart; i++)
    if(Sp->P[i].getType() == 4)
      local_star_count++;

  // Sum across all processes
  int total_star_count;
  MPI_Reduce(&local_star_count, &total_star_count, 1, MPI_INT, MPI_SUM, 0, Communicator);

  // Print only from task 0
  if(ThisTask == 0)
    SF_PRINT("STARFORMATION: %d total PartType4 stars exist at z=%.3f\n", 
          total_star_count, 1.0 / All.Time - 1.0);

   TIMER_STOP(CPU_COOLING_SFR);
 
 }
 
 /* FROM A NEWER GADGET-4 SF Ezra worked in */
 /** \brief Convert a SPH particle into a star.
  *
  *  This function converts an active star-forming gas particle into a star.
  *  The particle information of the gas is copied to the
  *  location istar and the fields necessary for the creation of the star
  *  particle are initialized.
  *
  *  \param i index of the gas particle to be converted
  *  \param birthtime time of birth (in code units) of the stellar particle
  */
  void coolsfr::convert_sph_particle_into_star(simparticles *Sp, int i, double birthtime)
  {
    SF_PRINT("STAR: convert_sph_particle_into_star() called. Convert gas particle into star of mass %.3f!\n", Sp->P[i].getMass());
 
    Sp->P[i].setType(STAR_TYPE);
  #if NSOFTCLASSES > 1
    Sp->P[i].setSofteningClass(All.SofteningClassOfPartType[Sp->P[i].getType()]);
  #endif
  #ifdef INDIVIDUAL_GRAVITY_SOFTENING
    if(((1 << Sp->P[i].getType()) & (INDIVIDUAL_GRAVITY_SOFTENING)))
      Sp->P[i].setSofteningClass(Sp->get_softening_type_from_mass(Sp->P[i].getMass()));
  #endif
  
    Sp->TimeBinSfr[Sp->P[i].getTimeBinHydro()] -= Sp->SphP[i].Sfr;
  
    Sp->P[i].StellarAge = birthtime;
  
    return;
  }
 
 /** \brief Spawn a star particle from a SPH gas particle.
  *
  *  This function spawns a star particle from an active star-forming
  *  SPH gas particle. The particle information of the gas is copied to the
  *  location istar and the fields necessary for the creation of the star
  *  particle are initialized. The total mass of the gas particle is split
  *  between the newly spawned star and the gas particle.
  *  (This function is probably unecessary)
  *
  *  \param igas index of the gas cell from which the star is spawned
  *  \param birthtime time of birth (in code units) of the stellar particle
  *  \param istar index of the spawned stellar particle
  *  \param mass_of_star the mass of the spawned stellar particle
  */
  void coolsfr::spawn_star_from_sph_particle(simparticles *Sp, int igas, double birthtime, int istar, MyDouble mass_of_star)
  {

    SF_PRINT("[STAR DEBUG] Called spawn_star_from_sph_particle. Created star ID=%d from gas ID=%d at time=%.6f\n",
        Sp->P[istar].ID.get(), Sp->P[igas].ID.get(), All.Time);

    Sp->P[istar] = Sp->P[igas];
    Sp->P[istar].setType(STAR_TYPE);

    #if NSOFTCLASSES > 1
      Sp->P[istar].setSofteningClass(All.SofteningClassOfPartType[Sp->P[istar].getType()]);
    #endif
    #ifdef INDIVIDUAL_GRAVITY_SOFTENING
      if(((1 << Sp->P[istar].getType()) & (INDIVIDUAL_GRAVITY_SOFTENING)))
        Sp->P[istar].setSofteningClass(Sp->get_softening_type_from_mass(Sp->P[istar].getMass()));
    #endif
  
    Sp->TimeBinsGravity.ActiveParticleList[Sp->TimeBinsGravity.NActiveParticles++] = istar;
  
    Sp->TimeBinsGravity.timebin_add_particle(istar, igas, Sp->P[istar].TimeBinGrav, Sp->TimeBinSynchronized[Sp->P[istar].TimeBinGrav]);
  
    Sp->P[istar].setMass(mass_of_star);
  
    Sp->P[istar].StellarAge = birthtime;
  
    // Assign feedback portions of the mass
    Sp->P[istar].Mass_SNII = 0.10 * Sp->P[istar].getMass();
    Sp->P[istar].Mass_AGB  = 0.35 * Sp->P[istar].getMass();
    Sp->P[istar].Mass_SNIa = 0.02 * Sp->P[istar].getMass();
  
    /* now change the conserved quantities in the cell in proportion */
    double fac = (Sp->P[igas].getMass() - Sp->P[istar].getMass()) / Sp->P[igas].getMass();
  
    Sp->P[igas].setMass(fac * Sp->P[igas].getMass());
  
    return;
  }
 
 
 
 
 #endif  /* STARFORMATION */
 #endif  /* COOLING */