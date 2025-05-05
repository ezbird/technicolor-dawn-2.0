/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file cooling.cc
 *
 *  \brief Module for gas radiative cooling
 */

#include "gadgetconfig.h"

#ifdef COOLING

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

#define eV_to_K 11604.505    // Conversion factor from eV to Kelvin
#define eV_to_erg 1.602e-12  // Conversion factor from eV to erg

// ----------------------------------------------------------------------
// Allocate the rate table for NCOOLTAB+1 entries
// ----------------------------------------------------------------------
void coolsfr::InitCoolMemory()
{
    // Allocate one contiguous array of rate_table structs
    RateT = (rate_table *)Mem.mymalloc("RateT", (NCOOLTAB + 1) * sizeof(rate_table));
}

/** \brief Compute the new internal energy per unit mass.
 *
 *   The function solves for the new internal energy per unit mass of the gas by integrating the equation
 *   for the internal energy with an implicit Euler scheme. The root of resulting non linear equation,
 *   which gives tnew internal energy, is found with the bisection method.
 *   Arguments are passed in code units.
 *
 *   \param u_old the initial (before cooling is applied) internal energy per unit mass of the gas particle
 *   \param rho   the proper density of the gas particle
 *   \param dt    the duration of the time step
 *   \param ne_guess electron number density relative to hydrogen number density (for molecular weight computation)
 *   \return the new internal energy per unit mass of the gas particle
 */
 double coolsfr::DoCooling(double u_old, double rho, double dt, double *ne_guess, gas_state *gs, const do_cool_data *DoCool)
{
    do_cool_data localDoCool;
    localDoCool.u_old_input = u_old;
    localDoCool.rho_input = rho;
    localDoCool.dt_input = dt;
    localDoCool.ne_guess_input = *ne_guess;

  if(!gsl_finite(u_old))
    Terminate("invalid input: u_old=%g\n", u_old);

  rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam; /* convert to physical cgs units */
  u_old *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
  dt *= All.UnitTime_in_s / All.HubbleParam;

  gs->nHcgs       = gs->XH * rho / PROTONMASS; /* hydrogen number dens in cgs units */
  double ratefact = gs->nHcgs * gs->nHcgs / rho;

  double u       = u_old;
  double u_lower = u;
  double u_upper = u;

  double LambdaNet = CoolingRateFromU(u, rho, ne_guess, gs, &localDoCool);

  /* bracketing */

  if(u - u_old - ratefact * LambdaNet * dt < 0) /* heating */
    {
      u_upper *= sqrt(1.1);
      u_lower /= sqrt(1.1);
      while(u_upper - u_old - ratefact * CoolingRateFromU(u_upper, rho, ne_guess, gs, &localDoCool) * dt < 0)
        {
          u_upper *= 1.1;
          u_lower *= 1.1;
        }
    }

  if(u - u_old - ratefact * LambdaNet * dt > 0)
    {
      u_lower /= sqrt(1.1);
      u_upper *= sqrt(1.1);
      while(u_lower - u_old - ratefact * CoolingRateFromU(u_lower, rho, ne_guess, gs, &localDoCool) * dt > 0)
        {
          u_upper /= 1.1;
          u_lower /= 1.1;
        }
    }

  int iter = 0;
  double du;
  do
    {
      u = 0.5 * (u_lower + u_upper);

      LambdaNet = CoolingRateFromU(u, rho, ne_guess, gs, &localDoCool);

      if(u - u_old - ratefact * LambdaNet * dt > 0)
        {
          u_upper = u;
        }
      else
        {
          u_lower = u;
        }

      du = u_upper - u_lower;

      iter++;

      if(iter >= (MAXITER - 10))
        printf("u= %g\n", u);
    }
  while(fabs(du / u) > 1.0e-6 && iter < MAXITER);

  if(iter >= MAXITER)
    Terminate(
        "failed to converge in DoCooling(): DoCool->u_old_input=%g\nDoCool->rho_input= %g\nDoCool->dt_input= "
        "%g\nDoCool->ne_guess_input= %g\n",
        localDoCool.u_old_input, localDoCool.rho_input, localDoCool.dt_input, localDoCool.ne_guess_input);

  u *= All.UnitDensity_in_cgs / All.UnitPressure_in_cgs; /* to internal units */

  return u;
}

/** \brief Return the cooling time.
 *
 *  If we actually have heating, a cooling time of 0 is returned.
 *
 *  \param u_old the initial (before cooling is applied) internal energy per unit mass of the gas particle
 *  \param rho   the proper density of the gas particle
 *  \param ne_guess electron number density relative to hydrogen number density (for molecular weight computation)
 */
double coolsfr::GetCoolingTime(double u_old, double rho, double *ne_guess, gas_state *gs, const do_cool_data *DoCool)
{
    do_cool_data localDoCool;
    localDoCool.u_old_input = u_old;
    localDoCool.rho_input = rho;
    localDoCool.ne_guess_input = *ne_guess;

  rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam; /* convert to physical cgs units */
  u_old *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;

  gs->nHcgs       = gs->XH * rho / PROTONMASS; /* hydrogen number dens in cgs units */
  double ratefact = gs->nHcgs * gs->nHcgs / rho;

  double u = u_old;

  double LambdaNet = CoolingRateFromU(u, rho, ne_guess, gs, &localDoCool);

  /* bracketing */

  if(LambdaNet >= 0) /* ups, we have actually heating due to UV background */
    return 0;

  double coolingtime = u_old / (-ratefact * LambdaNet);

  coolingtime *= All.HubbleParam / All.UnitTime_in_s;

  return coolingtime;
}

/** \brief Compute gas temperature from internal energy per unit mass.
 *
 *   This function determines the electron fraction, and hence the mean
 *   molecular weight. With it arrives at a self-consistent temperature.
 *   Element abundances and the rates for the emission are also computed
 *
 *  \param u   internal energy per unit mass
 *  \param rho gas density
 *  \param ne_guess electron number density relative to hydrogen number density
 *  \return the gas temperature
 */
 double coolsfr::convert_u_to_temp(double u, double rho, double *ne_guess,
  gas_state *gs, const do_cool_data *DoCool)
{
// --- Convert code‐units of u (erg/g in code) to physical cgs units ---
//   u_code × (UnitEnergy_in_cgs / UnitMass_in_g) → u_cgs [erg/g]
double u_cgs = u * (All.UnitEnergy_in_cgs / All.UnitMass_in_g);

// Handle non‐positive internal energy
if(u_cgs <= 0.0) {
*ne_guess = 0.0;
return 1.0;  // floor at 1 K
}

// Initial guess for μ = mean molecular weight
double mu = (1.0 + 4.0*gs->yhelium) / (1.0 + gs->yhelium + *ne_guess);

// Initial temperature estimate in Kelvin:
//   (γ−1) u [erg/g] × (m_p/k_B) × μ → T [K]
double temp = GAMMA_MINUS1 * u_cgs * PROTONMASS / BOLTZMANN * mu;

// enforce physical bounds on the guess
temp = std::min(std::max(temp, 1.0), 1e12);

// solver parameters
const int    MAX_ITERS      = 50;
const double REL_TOL       = 1e-3;    // relative tolerance
const double ABS_TOL       = 1e-6;    // absolute tolerance in K

double temp_old = temp;
for(int iter = 0; iter < MAX_ITERS; ++iter) {
double ne_old = *ne_guess;

// update electron fraction & rates at current T
find_abundances_and_rates(std::log10(temp), rho, ne_guess, gs, DoCool);

// recompute μ with updated ne
mu = (1.0 + 4.0*gs->yhelium) / (1.0 + gs->yhelium + *ne_guess);

// new temperature from updated μ
double temp_new = GAMMA_MINUS1 * u_cgs * PROTONMASS / BOLTZMANN * mu;

// damp oscillations lightly
double damping = 1.0 + 0.1 * iter;
temp = temp_old + (temp_new - temp_old)/damping;

// clamp to physical range each iteration
temp = std::min(std::max(temp, 1.0), 1e12);

// convergence check: either relative or absolute
double dt = std::abs(temp - temp_old);
if(dt < std::max(REL_TOL*temp, ABS_TOL)) {
return temp; 
}
temp_old = temp;
}

// if we get here, we failed to converge – warn and return last value
printf("convert_u_to_temp did not converge in %d iterations, T≈%g K", MAX_ITERS, temp);
return temp;
}


/** \brief Computes the actual abundance ratios.
 *
 *  The chemical composition of the gas is primordial (no metals are present)
 *
 *  \param logT     log10 of gas temperature
 *  \param rho      gas density
 *  \param ne_guess electron number density relative to hydrogen number density
 */
void coolsfr::find_abundances_and_rates(double logT, double rho, double *ne_guess, gas_state *gs, const do_cool_data *DoCool)
{
  double logT_input = logT;
  double rho_input  = rho;
  double ne_input   = *ne_guess;

  if(!gsl_finite(logT))
    Terminate("logT=%g\n", logT);

  if(logT <= Tmin) /* everything neutral */
    {
      gs->nH0   = 1.0;
      gs->nHe0  = gs->yhelium;
      gs->nHp   = 0;
      gs->nHep  = 0;
      gs->nHepp = 0;
      gs->ne    = 0;
      *ne_guess = 0;
      return;
    }

  if(logT >= Tmax) /* everything is ionized */
    {
      gs->nH0   = 0;
      gs->nHe0  = 0;
      gs->nHp   = 1.0;
      gs->nHep  = 0;
      gs->nHepp = gs->yhelium;
      gs->ne    = gs->nHp + 2.0 * gs->nHepp;
      *ne_guess = gs->ne; /* note: in units of the hydrogen number density */
      return;
    }

  double t    = (logT - Tmin) / deltaT;
  int j       = (int)t;
  double fhi  = t - j;
  double flow = 1 - fhi;

  if(*ne_guess == 0)
    *ne_guess = 1.0;

  gs->nHcgs = gs->XH * rho / PROTONMASS; /* hydrogen number dens in cgs units */

  gs->ne       = *ne_guess;
  double neold = gs->ne;
  int niter    = 0;
  gs->necgs    = gs->ne * gs->nHcgs;

  /* evaluate number densities iteratively (cf KWH eqns 33-38) in units of nH */
  do
    {
      niter++;

      gs->aHp   = flow * RateT[j].AlphaHp + fhi * RateT[j + 1].AlphaHp;
      gs->aHep  = flow * RateT[j].AlphaHep + fhi * RateT[j + 1].AlphaHep;
      gs->aHepp = flow * RateT[j].AlphaHepp + fhi * RateT[j + 1].AlphaHepp;
      gs->ad    = flow * RateT[j].Alphad + fhi * RateT[j + 1].Alphad;
      gs->geH0  = flow * RateT[j].GammaeH0 + fhi * RateT[j + 1].GammaeH0;
      gs->geHe0 = flow * RateT[j].GammaeHe0 + fhi * RateT[j + 1].GammaeHe0;
      gs->geHep = flow * RateT[j].GammaeHep + fhi * RateT[j + 1].GammaeHep;

      if(gs->necgs <= 1.e-25 || pc.J_UV == 0)
        {
          gs->gJH0ne = gs->gJHe0ne = gs->gJHepne = 0;
        }
      else
        {
          gs->gJH0ne  = pc.gJH0 / gs->necgs;
          gs->gJHe0ne = pc.gJHe0 / gs->necgs;
          gs->gJHepne = pc.gJHep / gs->necgs;
        }

      gs->nH0 = gs->aHp / (gs->aHp + gs->geH0 + gs->gJH0ne); /* eqn (33) */
      gs->nHp = 1.0 - gs->nH0;                               /* eqn (34) */

      if((gs->gJHe0ne + gs->geHe0) <= SMALLNUM) /* no ionization at all */
        {
          gs->nHep  = 0.0;
          gs->nHepp = 0.0;
          gs->nHe0  = gs->yhelium;
        }
      else
        {
          gs->nHep = gs->yhelium /
                     (1.0 + (gs->aHep + gs->ad) / (gs->geHe0 + gs->gJHe0ne) + (gs->geHep + gs->gJHepne) / gs->aHepp); /* eqn (35) */
          gs->nHe0  = gs->nHep * (gs->aHep + gs->ad) / (gs->geHe0 + gs->gJHe0ne);                                     /* eqn (36) */
          gs->nHepp = gs->nHep * (gs->geHep + gs->gJHepne) / gs->aHepp;                                               /* eqn (37) */
        }

      neold = gs->ne;

      gs->ne    = gs->nHp + gs->nHep + 2 * gs->nHepp; /* eqn (38) */
      gs->necgs = gs->ne * gs->nHcgs;

      if(pc.J_UV == 0)
        break;

      double nenew = 0.5 * (gs->ne + neold);
      gs->ne       = nenew;
      gs->necgs    = gs->ne * gs->nHcgs;

      if(fabs(gs->ne - neold) < 1.0e-4)
        break;

      if(niter > (MAXITER - 10))
        printf("ne= %g  niter=%d\n", gs->ne, niter);
    }
  while(niter < MAXITER);

  if(niter >= MAXITER)
    Terminate(
        "no convergence reached in find_abundances_and_rates(): logT_input= %g  rho_input= %g  ne_input= %g "
        "DoCool->u_old_input=%g\nDoCool->rho_input= %g\nDoCool->dt_input= %g\nDoCool->ne_guess_input= %g\n",
        logT_input, rho_input, ne_input, DoCool->u_old_input, DoCool->rho_input, DoCool->dt_input, DoCool->ne_guess_input);

  gs->bH0  = flow * RateT[j].BetaH0 + fhi * RateT[j + 1].BetaH0;
  gs->bHep = flow * RateT[j].BetaHep + fhi * RateT[j + 1].BetaHep;
  gs->bff  = flow * RateT[j].Betaff + fhi * RateT[j + 1].Betaff;

  *ne_guess = gs->ne;
}

/** \brief Get cooling rate from gas internal energy.
 *
 *  This function first computes the self-consistent temperature
 *  and abundance ratios, and then it calculates
 *  (heating rate-cooling rate)/n_h^2 in cgs units
 *
 *  \param u   gas internal energy per unit mass
 *  \param rho gas density
 *  \param ne_guess electron number density relative to hydrogen number density
 */
double coolsfr::CoolingRateFromU(double u, double rho, double *ne_guess, gas_state *gs, const do_cool_data *DoCool)
{
  double temp = convert_u_to_temp(u, rho, ne_guess, gs, DoCool);

  return CoolingRate(log10(temp), rho, ne_guess, gs, DoCool);
}

/** \brief  This function computes the self-consistent temperature and abundance ratios.
 *
 *  Used only in the file io.c (maybe it is not necessary)
 *
 *  \param u   internal energy per unit mass
 *  \param rho gas density
 *  \param ne_guess electron number density relative to hydrogen number density
 *  \param nH0_pointer pointer to the neutral hydrogen fraction (set to current value in the GasState struct)
 *  \param nHeII_pointer pointer to the ionised helium fraction (set to current value in the GasState struct)
 */
double coolsfr::AbundanceRatios(double u, double rho, double *ne_guess, double *nH0_pointer, double *nHeII_pointer)
{
  gas_state gs          = GasState;
  do_cool_data DoCool   = DoCoolData;
  DoCool.u_old_input    = u;
  DoCool.rho_input      = rho;
  DoCool.ne_guess_input = *ne_guess;

  rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam; /* convert to physical cgs units */
  u *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;

  double temp = convert_u_to_temp(u, rho, ne_guess, &gs, &DoCool);

  *nH0_pointer   = gs.nH0;
  *nHeII_pointer = gs.nHep;

  return temp;
}

/** \brief  Calculate (heating rate-cooling rate)/n_h^2 in cgs units.
 *
 *  \param logT     log10 of gas temperature
 *  \param rho      gas density
 *  \param nelec    electron number density relative to hydrogen number density
 *  \return         (heating rate-cooling rate)/n_h^2
 */
double coolsfr::CoolingRate(double logT, double rho, double *nelec, gas_state *gs, const do_cool_data *DoCool)
{
  double Lambda, Heat;

  if(logT <= Tmin)
    logT = Tmin + 0.5 * deltaT; /* floor at Tmin */

  gs->nHcgs = gs->XH * rho / PROTONMASS; /* hydrogen number dens in cgs units */

  if(logT < Tmax)
    {
      find_abundances_and_rates(logT, rho, nelec, gs, DoCool);

      /* Compute cooling and heating rate (cf KWH Table 1) in units of nH**2 */
      double T = pow(10.0, logT);

      double LambdaExcH0   = gs->bH0 * gs->ne * gs->nH0;
      double LambdaExcHep  = gs->bHep * gs->ne * gs->nHep;
      double LambdaExc     = LambdaExcH0 + LambdaExcHep; /* excitation */
      double LambdaIonH0   = 2.18e-11 * gs->geH0 * gs->ne * gs->nH0;
      double LambdaIonHe0  = 3.94e-11 * gs->geHe0 * gs->ne * gs->nHe0;
      double LambdaIonHep  = 8.72e-11 * gs->geHep * gs->ne * gs->nHep;
      double LambdaIon     = LambdaIonH0 + LambdaIonHe0 + LambdaIonHep; /* ionization */
      double LambdaRecHp   = 1.036e-16 * T * gs->ne * (gs->aHp * gs->nHp);
      double LambdaRecHep  = 1.036e-16 * T * gs->ne * (gs->aHep * gs->nHep);
      double LambdaRecHepp = 1.036e-16 * T * gs->ne * (gs->aHepp * gs->nHepp);
      double LambdaRecHepd = 6.526e-11 * gs->ad * gs->ne * gs->nHep;
      double LambdaRec     = LambdaRecHp + LambdaRecHep + LambdaRecHepp + LambdaRecHepd;
      double LambdaFF      = gs->bff * (gs->nHp + gs->nHep + 4 * gs->nHepp) * gs->ne;
      Lambda               = LambdaExc + LambdaIon + LambdaRec + LambdaFF;

      if(All.ComovingIntegrationOn)
        {
          double redshift    = 1 / All.Time - 1;
          double LambdaCmptn = 5.65e-36 * gs->ne * (T - 2.73 * (1. + redshift)) * pow(1. + redshift, 4.) / gs->nHcgs;

          Lambda += LambdaCmptn;
        }

      Heat = 0;
      if(pc.J_UV != 0)
        Heat += (gs->nH0 * pc.epsH0 + gs->nHe0 * pc.epsHe0 + gs->nHep * pc.epsHep) / gs->nHcgs;
    }
  else /* here we're outside of tabulated rates, T>Tmax K */
    {
      /* at high T (fully ionized); only free-free and Compton cooling are present. Assumes no heating. */

      Heat = 0;

      /* very hot: H and He both fully ionized */
      gs->nHp   = 1.0;
      gs->nHep  = 0;
      gs->nHepp = gs->yhelium;
      gs->ne    = gs->nHp + 2.0 * gs->nHepp;
      *nelec    = gs->ne; /* note: in units of the hydrogen number density */

      double T        = pow(10.0, logT);
      double LambdaFF = 1.42e-27 * sqrt(T) * (1.1 + 0.34 * exp(-(5.5 - logT) * (5.5 - logT) / 3)) * (gs->nHp + 4 * gs->nHepp) * gs->ne;
      double LambdaCmptn;
      if(All.ComovingIntegrationOn)
        {
          double redshift = 1 / All.Time - 1;
          /* add inverse Compton cooling off the microwave background */
          LambdaCmptn = 5.65e-36 * gs->ne * (T - 2.73 * (1. + redshift)) * pow(1. + redshift, 4.) / gs->nHcgs;
        }
      else
        LambdaCmptn = 0;

      Lambda = LambdaFF + LambdaCmptn;
    }

  return (Heat - Lambda);
}

/** \brief Make cooling rates interpolation table.
 *
 *  Set up interpolation tables in T for cooling rates given in KWH, ApJS, 105, 19
 */
void coolsfr::MakeRateTable(void)
{
  GasState.yhelium = (1 - GasState.XH) / (4 * GasState.XH);
  GasState.mhboltz = PROTONMASS / BOLTZMANN;

  deltaT          = (Tmax - Tmin) / NCOOLTAB;
  GasState.ethmin = pow(10.0, Tmin) * (1. + GasState.yhelium) / ((1. + 4. * GasState.yhelium) * GasState.mhboltz * GAMMA_MINUS1);
  /* minimum internal energy for neutral gas */

  for(int i = 0; i <= NCOOLTAB; i++)
    {
      RateT[i].BetaH0 = RateT[i].BetaHep = RateT[i].Betaff = RateT[i].AlphaHp = RateT[i].AlphaHep = RateT[i].AlphaHepp =
          RateT[i].Alphad = RateT[i].GammaeH0 = RateT[i].GammaeHe0 = RateT[i].GammaeHep = 0;

      double T     = pow(10.0, Tmin + deltaT * i);
      double Tfact = 1.0 / (1 + sqrt(T / 1.0e5));

      /* collisional excitation */
      /* Cen 1992 */
      if(118348 / T < 70)
        RateT[i].BetaH0 = 7.5e-19 * exp(-118348 / T) * Tfact;
      if(473638 / T < 70)
        RateT[i].BetaHep = 5.54e-17 * pow(T, -0.397) * exp(-473638 / T) * Tfact;

      /* free-free */
      RateT[i].Betaff = 1.43e-27 * sqrt(T) * (1.1 + 0.34 * exp(-(5.5 - log10(T)) * (5.5 - log10(T)) / 3));

      /* recombination */

      /* Cen 1992 */
      /* Hydrogen II */
      RateT[i].AlphaHp = 8.4e-11 * pow(T / 1000, -0.2) / (1. + pow(T / 1.0e6, 0.7)) / sqrt(T);
      /* Helium II */
      RateT[i].AlphaHep = 1.5e-10 * pow(T, -0.6353);
      /* Helium III */
      RateT[i].AlphaHepp = 4. * RateT[i].AlphaHp;
      /* Cen 1992 */
      /* dielectric recombination */
      if(470000 / T < 70)
        RateT[i].Alphad = 1.9e-3 * pow(T, -1.5) * exp(-470000 / T) * (1. + 0.3 * exp(-94000 / T));

      /* collisional ionization */
      /* Cen 1992 */
      /* Hydrogen */
      if(157809.1 / T < 70)
        RateT[i].GammaeH0 = 5.85e-11 * sqrt(T) * exp(-157809.1 / T) * Tfact;
      /* Helium */
      if(285335.4 / T < 70)
        RateT[i].GammaeHe0 = 2.38e-11 * sqrt(T) * exp(-285335.4 / T) * Tfact;
      /* Hellium II */
      if(631515.0 / T < 70)
        RateT[i].GammaeHep = 5.68e-12 * sqrt(T) * exp(-631515.0 / T) * Tfact;
    }
}

/** \brief Read table input for ionizing parameters.
 *
 *  \param file that contains the tabulated parameters
 */
void coolsfr::ReadIonizeParams(char *fname)
{
  NheattabUVB = 0;
  int i, iter;
  for(iter = 0, i = 0; iter < 2; iter++)
    {
      FILE *fdcool;
      if(!(fdcool = fopen(fname, "r")))
        Terminate(" Cannot read ionization table in file `%s'\n", fname);
      if(iter == 0)
        while(fscanf(fdcool, "%*g %*g %*g %*g %*g %*g %*g") != EOF)
          NheattabUVB++;
      if(iter == 1)
        while(fscanf(fdcool, "%g %g %g %g %g %g %g", &PhotoTUVB[i].variable, &PhotoTUVB[i].gH0, &PhotoTUVB[i].gHe, &PhotoTUVB[i].gHep,
                     &PhotoTUVB[i].eH0, &PhotoTUVB[i].eHe, &PhotoTUVB[i].eHep) != EOF)
          i++;
      fclose(fdcool);

      if(iter == 0)
        {
          PhotoTUVB = (photo_table *)Mem.mymalloc("PhotoT", NheattabUVB * sizeof(photo_table));
          mpi_printf("COOLING: read ionization table with %d entries in file `%s'.\n", NheattabUVB, fname);
        }
    }
  /* ignore zeros at end of treecool file */
  for(i = 0; i < NheattabUVB; ++i)
    if(PhotoTUVB[i].gH0 == 0.0)
      break;

  NheattabUVB = i;
  mpi_printf("COOLING: using %d ionization table entries from file `%s'.\n", NheattabUVB, fname);

  if(NheattabUVB < 1)
    Terminate("The length of the cooling table has to have at least one entry");
}

/** \brief Set the ionization parameters for the UV background.
 */
void coolsfr::IonizeParamsUVB(void)
{
  if(!All.ComovingIntegrationOn)
    {
      SetZeroIonization();
      return;
    }

  if(NheattabUVB == 1)
    {
      /* treat the one value given as constant with redshift */
      pc.J_UV   = 1;
      pc.gJH0   = PhotoTUVB[0].gH0;
      pc.gJHe0  = PhotoTUVB[0].gHe;
      pc.gJHep  = PhotoTUVB[0].gHep;
      pc.epsH0  = PhotoTUVB[0].eH0;
      pc.epsHe0 = PhotoTUVB[0].eHe;
      pc.epsHep = PhotoTUVB[0].eHep;
    }
  else
    {
      double redshift = 1 / All.Time - 1;
      double logz     = log10(redshift + 1.0);
      int ilow        = 0;
      for(int i = 0; i < NheattabUVB; i++)
        {
          if(PhotoTUVB[i].variable < logz)
            ilow = i;
          else
            break;
        }

      if(logz > PhotoTUVB[NheattabUVB - 1].variable || ilow >= NheattabUVB - 1)
        {
          SetZeroIonization();
        }
      else
        {
          double dzlow = logz - PhotoTUVB[ilow].variable;
          double dzhi  = PhotoTUVB[ilow + 1].variable - logz;

          if(PhotoTUVB[ilow].gH0 == 0 || PhotoTUVB[ilow + 1].gH0 == 0)
            {
              SetZeroIonization();
            }
          else
            {
              pc.J_UV   = 1;
              pc.gJH0   = pow(10., (dzhi * log10(PhotoTUVB[ilow].gH0) + dzlow * log10(PhotoTUVB[ilow + 1].gH0)) / (dzlow + dzhi));
              pc.gJHe0  = pow(10., (dzhi * log10(PhotoTUVB[ilow].gHe) + dzlow * log10(PhotoTUVB[ilow + 1].gHe)) / (dzlow + dzhi));
              pc.gJHep  = pow(10., (dzhi * log10(PhotoTUVB[ilow].gHep) + dzlow * log10(PhotoTUVB[ilow + 1].gHep)) / (dzlow + dzhi));
              pc.epsH0  = pow(10., (dzhi * log10(PhotoTUVB[ilow].eH0) + dzlow * log10(PhotoTUVB[ilow + 1].eH0)) / (dzlow + dzhi));
              pc.epsHe0 = pow(10., (dzhi * log10(PhotoTUVB[ilow].eHe) + dzlow * log10(PhotoTUVB[ilow + 1].eHe)) / (dzlow + dzhi));
              pc.epsHep = pow(10., (dzhi * log10(PhotoTUVB[ilow].eHep) + dzlow * log10(PhotoTUVB[ilow + 1].eHep)) / (dzlow + dzhi));
            }
        }
    }
}

/** \brief Reset the ionization parameters.
 */
void coolsfr::SetZeroIonization(void) { memset(&pc, 0, sizeof(photo_current)); }

/** \brief Wrapper function to set the ionizing background.
 */
void coolsfr::IonizeParams(void) { IonizeParamsUVB(); }

/** \brief Initialize the cooling module.
 *
 *   This function initializes the cooling module. In particular,
 *   it allocates the memory for the cooling rate and ionization tables
 *   and initializes them.
 */
/** \brief Initialize the cooling module by loading TREECOOL tables */
void coolsfr::InitCool(void)
{
    GasState.XH = HYDROGEN_MASSFRAC;
    SetZeroIonization();

    // — allocate the table and fill it with TREECOOL data —
    InitCoolMemory();       // <— your new allocator
    MakeCoolingTable();     // <— this should now write into RateT[0…NCOOLTAB]

    // — UV/photo-heating as before —
    ReadIonizeParams(All.TreecoolFile);
    All.Time = All.TimeBegin;
    All.set_cosmo_factors_for_current_time();
    IonizeParams();
}

void coolsfr::MakeCoolingTable()
{
    // — set primordial composition and constants —
    GasState.XH      = HYDROGEN_MASSFRAC;
    GasState.yhelium = (1.0 - GasState.XH) / (4.0 * GasState.XH);
    GasState.mhboltz = PROTONMASS / BOLTZMANN;

    // — compute log-T spacing and minimum internal energy —
    deltaT = (Tmax - Tmin) / double(NCOOLTAB);
    GasState.ethmin = std::pow(10.0, Tmin)
                    * (1.0 + GasState.yhelium)
                    / ((1.0 + 4.0 * GasState.yhelium)
                       * GasState.mhboltz
                       * GAMMA_MINUS1);

    // — fill the table —
    for(int i = 0; i <= NCOOLTAB; i++)
    {
        double logT = Tmin + deltaT * i;
        double T    = std::pow(10.0, logT);
        double tfac = 1.0 / (1.0 + std::sqrt(T / 1e5));

        // H⁰ collisional ionization (Scholz–Walters ’91)
        double betaH0;
        if(T >= 2e3 && T < 1e8)
        {
            double b0,b1,b2,b3,b4,b5;
            double c0,c1,c2,c3,c4,c5;
            if(T < 6e4)
            {
                b0=-3.299613e1; b1=1.858848e1; b2=-6.052265;
                b3=8.603783e-1; b4=-5.717760e-2; b5=1.451330e-3;
                c0=-1.630155e2; c1=8.795711e1; c2=-2.057117e1;
                c3=2.359573e0; c4=-1.339059e-1; c5=3.021507e-3;
            }
            else if(T < 6e6)
            {
                b0=2.869759e2;  b1=-1.077956e2; b2=1.524107e1;
                b3=-1.080538e0; b4=3.836975e-2; b5=-5.467273e-4;
                c0=5.279996e2;  c1=-1.939399e2; c2=2.718982e1;
                c3=-1.883399e0; c4=6.462462e-2; c5=-8.811076e-4;
            }
            else
            {
                b0=-2.7604708e3; b1=7.9339351e2;  b2=-9.1198462e1;
                b3=5.1993362e0;  b4=-1.4685343e-1; b5=1.6404093e-3;
                c0=-2.8133632e3; c1=8.1509685e2;  c2=-9.4418414e1;
                c3=5.4280565e0;  c4=-1.5467120e-1; c5=1.7439112e-3;
            }
            double y   = std::log(T);
            double E2  = 10.2; // eV
            double g2s = std::exp(b0 + b1*y + b2*y*y + b3*y*y*y + b4*y*y*y*y + b5*y*y*y*y*y);
            double g2p = std::exp(c0 + c1*y + c2*y*y + c3*y*y*y + c4*y*y*y*y + c5*y*y*y*y*y);
            double TeV = T / eV_to_K;
            betaH0     = E2 * eV_to_erg * (g2s + g2p) * std::exp(-E2 / TeV);
        }
        else
        {
            betaH0 = (118348.0/T < 70.0)
                   ? 7.5e-19 * std::exp(-118348.0/T) * tfac
                   : 0.0;
        }

        // He⁺ collisional ionization
        double betaHep = (473638.0/T < 70.0)
                       ? 5.54e-17 * std::pow(T, -0.397) * std::exp(-473638.0/T) * tfac
                       : 0.0;

        // free–free cooling
        double betaff = 1.43e-27 * std::sqrt(T)
                      * (1.1 + 0.34 * std::exp(-std::pow(5.5 - std::log10(T),2)/3.0));

        // radiative recombination
        double alphaHp   = 6.28e-11
                         * std::pow(T/1000.0, -0.2)
                         / (1.0 + std::pow(T/1e6, 0.7))
                         / std::sqrt(T);
        double alphaHep  = 1.5e-10 * std::pow(T, -0.6353);
        double alphaHepp = 3.36e-10
                         * std::pow(T/1000.0, -0.2)
                         / (1.0 + std::pow(T/4e6, 0.7))
                         / std::sqrt(T);

        // dielectronic recombination
        double alphad = (470000.0/T < 70.0)
                      ? 1.9e-3 * std::pow(T, -1.5) * std::exp(-470000.0/T)
                        * (1.0 + 0.3 * std::exp(-94000.0/T))
                      : 0.0;

        // high-T collisional ionization (Voronov ’97)
        double TeV = T / eV_to_K;
        double U;

        U = 13.6/TeV;
        double geH0  = 0.291e-7 * std::pow(U, 0.39) * std::exp(-U) / (0.232 + U);
        U = 24.6/TeV;
        double geHe0 = 0.175e-7 * std::pow(U, 0.35) * std::exp(-U) / (0.18 + U);
        U = 54.4/TeV;
        double geHep = 0.205e-8 * (1.0 + std::sqrt(U)) * std::pow(U, 0.25)
                      * std::exp(-U) / (0.265 + U);

        // store into the table
        RateT[i].BetaH0    = betaH0;
        RateT[i].BetaHep   = betaHep;
        RateT[i].Betaff    = betaff;
        RateT[i].AlphaHp   = alphaHp;
        RateT[i].AlphaHep  = alphaHep;
        RateT[i].Alphad    = alphad;
        RateT[i].AlphaHepp = alphaHepp;
        RateT[i].GammaeH0  = geH0;
        RateT[i].GammaeHe0 = geHe0;
        RateT[i].GammaeHep = geHep;
    }
}


/** \brief Apply the isochoric cooling to all the active gas particles.
 *
 */
 void coolsfr::cool_sph_particle(simparticles *Sp, int i, gas_state *gs, const do_cool_data *DoCool)
 {
   double dens = Sp->SphP[i].Density;
 
   double dt = (Sp->P[i].getTimeBinHydro() ? (((integertime)1) << Sp->P[i].getTimeBinHydro()) : 0) * All.Timebase_interval;
 
   double dtime = All.cf_atime * dt / All.cf_atime_hubble_a;
 
   double utherm = Sp->get_utherm_from_entropy(i);
 
   double ne = Sp->SphP[i].Ne; /* electron abundance (gives ionization state and mean molecular weight) */
   
   // Calculate minimum energy from MinGasTemp
   double mu = (1.0 + 4.0*gs->yhelium) / (1.0 + gs->yhelium + ne);
   double min_energy = All.MinGasTemp * BOLTZMANN / (GAMMA_MINUS1 * PROTONMASS * mu);
 
   // Use min_energy instead of All.MinEgySpec for the temperature floor
   double unew = DoCooling(std::max<double>(min_energy, utherm), dens * All.cf_a3inv, dtime, &ne, gs, DoCool);
   Sp->SphP[i].Ne = ne;
 
   if(unew < 0)
     Terminate("invalid temperature: i=%d unew=%g\n", i, unew);
 
   double du = unew - utherm;
 
   // Apply temperature floor based on MinGasTemp
   if(unew < min_energy)
     du = min_energy - utherm;
 
   utherm += du;
 
 #ifdef OUTPUT_COOLHEAT
   if(dtime > 0)
     Sp->SphP[i].CoolHeat = du * Sp->P[i].getMass() / dtime;
 #endif
 
   Sp->set_entropy_from_utherm(utherm, i);
   Sp->SphP[i].set_thermodynamic_variables();
 }

#endif