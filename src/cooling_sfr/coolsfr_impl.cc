/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file coolsfr_impl.cc
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

#define eV_to_K 11604.505  // Conversion factor from eV to Kelvin

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

#ifdef H2REGSF
      All.PhysDensThresh = (DENSTHRESH/(All.HubbleParam*All.HubbleParam)) * PROTONMASS / (HYDROGEN_MASSFRAC*All.UnitDensity_in_cgs);
      All.OTUVThresh     = ((DENSTHRESH*OTUVTHRESH)/(All.HubbleParam*All.HubbleParam)) * PROTONMASS / (HYDROGEN_MASSFRAC*All.UnitDensity_in_cgs);
      
      double pctocm = 3.085678e18;
      double gtomsun = 1.989e33;
      All.SIGMA_NORM = 1./((All.UnitMass_in_g/(All.UnitLength_in_cm*All.UnitLength_in_cm))*((pctocm*pctocm)/gtomsun));
      
      double s_to_gyr, G_internalunit, GyrInternal;
      s_to_gyr = 3.15569e16;
      G_internalunit = 6.674e-8 * All.UnitMass_in_g/(All.UnitLength_in_cm*All.UnitLength_in_cm*All.UnitLength_in_cm)*(s_to_gyr*s_to_gyr);
      GyrInternal    = 1./(0.98/All.HubbleParam);
      All.GInternal  = G_internalunit / (GyrInternal*GyrInternal);
      
      if(ThisTask == 0)
        {
          mpi_printf("COOLSFR: Setting PhysDensThresh=%g in pressure sfr (system unit) %g h^2 cm^-3\n", All.PhysDensThresh,
                     All.PhysDensThresh / (PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs));
          mpi_printf("COOLSFR: Setting OTUVThresh=%g (system unit) %g h^2 cm^-3\n", All.OTUVThresh,
                     All.OTUVThresh / (PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs));
          mpi_printf("COOLSFR: Setting SIGMA_NORM=%g (system unit)\n", All.SIGMA_NORM);
        }
#endif

#if defined(SGREGSF) 
      double s_to_gyr, G_internalunit, GyrInternal;
      s_to_gyr = 3.15569e16;
      G_internalunit = 6.674e-8 * All.UnitMass_in_g/(All.UnitLength_in_cm*All.UnitLength_in_cm*All.UnitLength_in_cm)*(s_to_gyr*s_to_gyr);
      GyrInternal    = 1./(0.98/All.HubbleParam);
      All.GInternal  = G_internalunit / (GyrInternal*GyrInternal);
      if(ThisTask==0)
        {
          mpi_printf("COOLSFR: All.G=%0.4e All.GInternal=%0.4e\n", All.G, All.GInternal);
        }
#endif

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
 * Calculate the star formation rate of a gas particle
 */
double coolsfr::get_starformation_rate(int i, double *xcloud, simparticles *Sp)
{
  double rateOfSF;
  double tsfr, cloudmass;
  gas_state gs = GasState;

#if defined(H2REGSF) || defined(H2AUTOSHIELD)
  double t_ff;
#endif

#ifdef SGREGSF
  double SG_numerator;
  double SG_alpha;
#endif

#if !defined(H2REGSF) && !defined(SGREGSF) && !defined(H2AUTOSHIELD)
  double factorEVP, egyhot, ne, tcool, y, x;
  do_cool_data DoCool = DoCoolData;
#endif

  *xcloud = 0.0; /* default: no molecular clouds */

#if defined(H2REGSF) || defined(H2AUTOSHIELD)
  if(Sp->SphP[i].fH2 > 0.0)
    Sp->SphP[i].SfFlag = 1;
#endif

#ifdef SGREGSF
  SG_numerator = (Sp->SphP[i].v.DivVel * Sp->SphP[i].v.DivVel) + (Sp->SphP[i].r.CurlVel * Sp->SphP[i].r.CurlVel);
  SG_alpha = SG_beta * SG_numerator / (Sp->SphP[i].Density * All.GInternal);

  if(SG_alpha < 1)
    Sp->SphP[i].SfFlag = 1;
#endif

  if(Sp->SphP[i].SfFlag == 0)
    return 0;

#ifdef H2REGSF
  t_ff = sqrt((3.0 * M_PI) / (32.0 * (Sp->SphP[i].Density * All.cf_a3inv) * All.G));
  tsfr = t_ff / H2REG_EFF;
  cloudmass = Sp->SphP[i].fH2 * Sp->SphP[i].XHI * Sp->SphP[i].nh0 * Sp->P[i].getMass();
  *xcloud = Sp->SphP[i].fH2;
#endif

#ifdef H2AUTOSHIELD
  t_ff = sqrt((3.0 * M_PI) / (32.0 * (Sp->SphP[i].Density * All.cf_a3inv) * All.G));
  tsfr = t_ff / H2REG_EFF;
  cloudmass = Sp->SphP[i].fH2 * HYDROGEN_MASSFRAC * Sp->P[i].getMass();
  *xcloud = Sp->SphP[i].fH2;
#endif

#ifdef SGREGSF
  tsfr = sqrt((3.0 * M_PI) / (32.0 * (Sp->SphP[i].Density * All.cf_a3inv) * All.GInternal));
  cloudmass = Sp->P[i].getMass();
  *xcloud = 1.0;
#endif

#if !defined(H2REGSF) && !defined(SGREGSF) && !defined(H2AUTOSHIELD)
  tsfr = sqrt(All.PhysDensThresh / (Sp->SphP[i].Density * All.cf_a3inv)) * All.MaxSfrTimescale;

  factorEVP = pow(Sp->SphP[i].Density * All.cf_a3inv / All.PhysDensThresh, -0.8) * All.FactorEVP;

  egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

  ne = Sp->SphP[i].Ne;
#ifdef TABLECOOL
  tcool = GetCoolingTime(egyhot, Sp->SphP[i].Density * All.cf_a3inv, &ne, &gs, &DoCool);
#else
  tcool = GetCoolingTime(egyhot, Sp->SphP[i].Density * All.cf_a3inv, &ne, &gs, &DoCool);
#endif

  y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
  x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));

  cloudmass = x * Sp->P[i].getMass();
  *xcloud = x;
#endif
  
  rateOfSF = cloudmass / tsfr;

  /* convert to solar masses per yr */
  rateOfSF *= (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

  return rateOfSF;
}

/**
 * Create a star particle from a gas particle
 */
void coolsfr::create_star_particle(simparticles *Sp, int i, double prob, double rnd)
{

    if(ThisTask == 0)
    {
      static int prob_checks = 0;
      if(prob_checks < 5)
      {
        printf("STARFORMATION: prob=%g, rnd=%g, ID=%llu\n", prob, rnd, Sp->P[i].ID.get());
        prob_checks++;
      }
    }

  if(rnd < prob)  /* Make a star */
    {
      if(Sp->P[i].getMass() < 1.5 * All.TargetGasMass / GENERATIONS)
        {
          // Convert the gas particle into a star
          Sp->P[i].setType(4);  // Change type to star
      
#ifdef STELLARAGE
          // Record stellar formation time
          Sp->P[i].StellarAge = All.Time;
#endif
        }
      else
        {
          // Spawn a new star particle
          double pmass = Sp->P[i].getMass();
      
          // Make sure we have enough memory
          if(Sp->NumPart + 1 > Sp->MaxPart)
            Terminate("COOLSFR: No space left for star particle creation");
      
          // Create new star particle in the last position
          int j = Sp->NumPart;
      
          Sp->P[j] = Sp->P[i];  // Copy properties from parent
          Sp->P[j].setType(4);  // Set type to star
      
          // Adjust masses
          Sp->P[j].setMass(All.TargetGasMass / GENERATIONS);
          Sp->P[i].setMass(pmass - Sp->P[j].getMass());
      
#ifdef STELLARAGE
          // Record stellar formation time
          Sp->P[j].StellarAge = All.Time;
#endif
          // Increment particle counter
          Sp->NumPart++;
        }
    }
}

/**
 * This function handles gas cooling and star formation
 */
void coolsfr::cooling_and_starformation(simparticles *Sp)
{
  TIMER_START(CPU_COOLING_SFR);
  
  All.set_cosmo_factors_for_current_time();
  
  double time_h_a = (All.ComovingIntegrationOn) ? All.Time * All.cf_hubble_a : 1.0;
  double total_sfr = 0;  // Total star formation rate
  int sf_eligible = 0;

  gas_state gs = GasState;
  do_cool_data DoCool = DoCoolData;
  
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
          
#if !defined(H2REGSF) && !defined(SGREGSF) && !defined(H2AUTOSHIELD)
          // Standard density-threshold star formation
          if(convert_u_to_temp(utherm, rho, &Sp->SphP[target].Ne, &gs, &DoCool) < All.TempSfrThresh) 
            {
              if(rho >= All.PhysDensThresh)
                Sp->SphP[target].SfFlag = 1;
                sf_eligible++;
            }
            
          // Additional cosmological threshold check
          if(All.ComovingIntegrationOn)
            if(Sp->SphP[target].Density < All.OverDensThresh)
              Sp->SphP[target].SfFlag = 0;
#endif

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

          // Process star-forming particles
          if(Sp->SphP[target].SfFlag == 1)
            {
              // Calculate cloud mass fraction
              double xcloud;
              double sfr = get_starformation_rate(target, &xcloud, Sp);
              Sp->SphP[target].Sfr = sfr;
              
              total_sfr += sfr;
              
                // DEBUGGING!
                if(ThisTask == 0)
                {
                    static int printed = 0;
                    if(printed < 5)
                    {
                    mpi_printf("STARFORMATION: particle %d, rho=%g, thresh=%g, xcloud=%g, sfr=%g\n", 
                                target, Sp->SphP[target].Density * All.cf_a3inv, All.PhysDensThresh, xcloud, sfr);
                    printed++;
                    }
                }

              // Calculate star formation probability
              double tsfr, factorEVP, egyhot, egyeff, egyold, egycurrent;
              double ne = Sp->SphP[target].Ne;
              
#if !defined(H2REGSF) && !defined(SGREGSF) && !defined(H2AUTOSHIELD)
              // Standard model
              tsfr = sqrt(All.PhysDensThresh / rho) * All.MaxSfrTimescale;
              factorEVP = pow(rho / All.PhysDensThresh, -0.8) * All.FactorEVP;
              egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;
#else
              // For H2 or SG models, use values from get_starformation_rate
              factorEVP = pow(rho / All.PhysDensThresh, -0.8) * All.FactorEVP;
              egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;
#endif

              double cloudmass = xcloud * Sp->P[target].getMass();
              if(tsfr < dtime) tsfr = dtime;
              
              // Probability of star formation
              double p = (1 - All.FactorSN) * dtime / tsfr * cloudmass / Sp->P[target].getMass();
              double prob;
              
              if(Sp->P[target].getMass() < 1.5 * All.TargetGasMass / GENERATIONS)
                prob = 1 - exp(-p);
              else
                prob = Sp->P[target].getMass() / (All.TargetGasMass / GENERATIONS) * (1 - exp(-p));
              
              // Apply effective equation of state for star-forming gas
              if(dt > 0)
                {
                  // Calculate energy from entropy
                  egycurrent = Sp->SphP[target].Entropy * pow(Sp->SphP[target].Density, GAMMA_MINUS1) / GAMMA_MINUS1;
                  
                  // Calculate relaxation timescale
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
              double rnd = get_random_number( Sp->P[target].ID.get() + target + All.NumCurrentTiStep );
              
              // Create star if probability check passes
              create_star_particle(Sp, target, prob, rnd);
              
#ifdef WINDS
              // Handle wind model if particle hasn't been turned into a star
              if(Sp->P[target].getType() == 0)
                spawn_wind_particle(Sp, target, cloudmass, tsfr);
#endif
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
  
  if(ThisTask == 0)
      mpi_printf("STARFORMATION: %d particles eligible for star formation\n", sf_eligible);

  // Output total star formation rate
  double glob_totsfr;
  MPI_Allreduce(&total_sfr, &glob_totsfr, 1, MPI_DOUBLE, MPI_SUM, Communicator);
  
  TIMER_STOP(CPU_COOLING_SFR);
}

/**
 * Helper function to get a random number for a given seed
 */
 double coolsfr::get_random_number(int id)
 {
   // Make sure ID is positive to avoid issues
   unsigned int posid = (id < 0) ? (unsigned int)(-id) : (unsigned int)(id);
   
   // Modulo to prevent integer overflow issues
   posid %= 1000000;
   
   int ia = 16807;
   int im = 2147483647;
   int iq = 127773;
   int ir = 2836;
   int iy, k;
   
   k = posid / iq;
   posid = ia * (posid - k * iq) - ir * k;
   if(posid < 0) posid += im;
   iy = posid;
   
   double dum = iy / (double)im;
   
   // For debugging - but without recursive call!
   if(ThisTask == 0)
   {
     static int rnd_checks = 0;
     if(rnd_checks < 5)
     {
       printf("STARFORMATION: random id=%d, result=%g\n", id, dum);
       rnd_checks++;
     }
   }
   
   return dum;
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


#endif /* STARFORMATION */
#endif /* COOLING */