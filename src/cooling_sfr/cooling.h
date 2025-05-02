/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file cooling.h
 *
 *  \brief defines a class for dealing with cooling and star formation
 */

 #ifndef COOLING_H
 #define COOLING_H
 
 #include "gadgetconfig.h"
 
 #ifdef COOLING
 
 #include "../data/simparticles.h"
 #include "../mpi_utils/setcomm.h"
 
 class coolsfr : public setcomm
 {
  public:
   coolsfr(MPI_Comm comm) : setcomm(comm) {}
 
   double AbundanceRatios(double u, double rho, double *ne_guess, double *nH0_pointer, double *nHeII_pointer);
 
   void InitCool(void);
   void IonizeParams(void);
 
   /**\n   * @brief Allocate and initialize the TREECOOL rate table
    */
   void InitCoolMemory();
 
   /**\n   * @brief Populate the TREECOOL rate table (wrapper to global MakeCoolingTable())
    */
   void MakeCoolingTable();
 
   void cooling_only(simparticles *Sp);
 
 #ifdef STARFORMATION
   void sfr_create_star_particles(simparticles *Sp);
   void set_units_sfr(void);
   void cooling_and_starformation(simparticles *Sp);
   void init_clouds(void);
 #endif
 
  private:
 #define NCOOLTAB 2000
 
   /* data for gas state */
   struct gas_state
   {
     double ne, necgs, nHcgs;
     double bH0, bHep, bff, aHp, aHep, aHepp, ad, geH0, geHe0, geHep;
     double gJH0ne, gJHe0ne, gJHepne;
     double nH0, nHp, nHep, nHe0, nHepp;
     double XH, yhelium;
     double mhboltz;
     double ethmin; /* minimum internal energy for neutral gas */
   };
 
   /* tabulated rates */
   struct rate_table
   {
     double BetaH0, BetaHep, Betaff;
     double AlphaHp, AlphaHep, Alphad, AlphaHepp;
     double GammaeH0, GammaeHe0, GammaeHep;
   };
 
   /* photo-ionization/heating rate table */
   struct photo_table
   {
     float variable;       /* logz for UVB */
     float gH0, gHe, gHep; /* photo-ionization rates */
     float eH0, eHe, eHep; /* photo-heating rates */
   };
 
   /* current interpolated photo-ionization/heating rates */
   struct photo_current
   {
     char J_UV;
     double gJH0, gJHep, gJHe0, epsH0, epsHep, epsHe0;
   };
 
   /* cooling data */
   struct do_cool_data
   {
     double u_old_input, rho_input, dt_input, ne_guess_input;
   };
 
   gas_state GasState;      /**< gas state */
   do_cool_data DoCoolData; /**< cooling data */
 
   rate_table *RateT;      /**< TREECOOL tabulated rates */
   photo_table *PhotoTUVB; /**< photo-ionization/heating rate table for UV background */
   photo_current pc;       /**< current interpolated photo rates */
 
   double Tmin = 1.0; /**< min temperature in log10 */
   double Tmax = 9.0; /**< max temperature in log10 */
   double deltaT;     /**< log10 of temperature spacing in the interpolation tables */
   int NheattabUVB;   /**< length of UVB photo table */
 
 #ifdef COOLING
   /* Core cooling routines */
   double DoCooling(double u_old, double rho, double dt, double *ne_guess,
                    gas_state *gs, const do_cool_data *DoCool);
   double GetCoolingTime(double u_old, double rho, double *ne_guess,
                         gas_state *gs, const do_cool_data *DoCool);
   void cool_sph_particle(simparticles *Sp, int i,
                          gas_state *gs, const do_cool_data *DoCool);
 
   void SetZeroIonization(void);
 #endif
 
   void integrate_sfr(void);
 
   /* Conversion and rate-finder routines */
   double CoolingRate(double logT, double rho, double *nelec,
                      gas_state *gs, const do_cool_data *DoCool);
   double CoolingRateFromU(double u, double rho, double *ne_guess,
                           gas_state *gs, const do_cool_data *DoCool);
   void find_abundances_and_rates(double logT, double rho,
                                  double *ne_guess, gas_state *gs,
                                  const do_cool_data *DoCool);
   void IonizeParamsUVB(void);
   void ReadIonizeParams(char *fname);
 
   /* Default G4 analytic table builder (guarded out by TABLECOOL) */
 #ifndef TABLECOOL
   void MakeRateTable(void);
 #endif
 };
 
 #endif /* COOLING */
 #endif /* COOLING_H */
 