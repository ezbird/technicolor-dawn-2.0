#include "gadgetconfig.h"
#ifdef COOLING

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>

#include "../cooling_sfr/cooling.h"
#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/mymalloc.h"
#include "../data/simparticles.h"
#include "../logs/logs.h"
#include "../logs/timer.h"
#include "../system/system.h"
#include "../time_integration/timestep.h"
#include "../data/constants.h"

// Define necessary global variables
#define JAMPL 1.0
#define TABLESIZE 500
#define NCOOLTAB 2000
#define XH 0.76
#define COOLLIM 0.01
#define COOL_TOL 0.002
#define MIN_DTIONFRAC 0.01
#define MAXITER 1000
#define SMALLNUM 1.0e-60
#define EQUILIBRIUM_IONIZED 0

// Cooling table definitions
#define RHO_TABLE_SIZE 200
#define TEMP_TABLE_SIZE 200
#define NRHOTAB 240
#define NTEMPTAB 900

// Globals for cooling
static double coolTable[RHO_TABLE_SIZE][TEMP_TABLE_SIZE];
static double neTable[RHO_TABLE_SIZE][TEMP_TABLE_SIZE];
static double Tmin = 1.0;
static double Tmax = 9.0;
static double RhoMin = -8.0;
static double RhoMax = 4.0;
static double dlogT, dlogRho;
static double yhelium;
static double mhboltz;

// Cooling function lookup tables
static double BetaH0[NCOOLTAB], BetaHep[NCOOLTAB], Betaff[NCOOLTAB];
static double AlphaHp[NCOOLTAB], AlphaHep[NCOOLTAB], AlphaHepp[NCOOLTAB], Alphad[NCOOLTAB];
static double GammaeH0[NCOOLTAB], GammaeHe0[NCOOLTAB], GammaeHep[NCOOLTAB];

// TREECOOL table variables
static double J_UV = 0, gJH0 = 0, gJHep = 0, gJHe0 = 0, epsH0 = 0, epsHep = 0, epsHe0 = 0;
static double redshift_old = -100., gJH0_old = -100.;

// Advanced cooling table variables
static float inlogz[TABLESIZE];   // log10(z+1)
static float gH0[TABLESIZE];      // photoionization rate for H0
static float gHe[TABLESIZE];      // photoionization rate for He0
static float gHep[TABLESIZE];     // photoionization rate for He+
static float eH0[TABLESIZE];      // photoheating rate for H0
static float eHe[TABLESIZE];      // photoheating rate for He0
static float eHep[TABLESIZE];     // photoheating rate for He+
static int nheattab = 0;          // actual length of the table

// Additional variables needed
static double rhomin, rhomax;
static double drhoinv, dTinv;
static double rhoarray[NRHOTAB], Tarray[NTEMPTAB];
static double nH0, nHe0, nHep, nHepp;
static int npcool = 0, itercool = 0;

// Function prototypes
void coolsfr::ReadIonizeParams(const char* fname);
void coolsfr::IonizeParams();
void coolsfr::IonizeParamsTable();
void coolsfr::MakeCoolingTable();
double coolsfr::DoCooling(double u_old, double rho, double dt, double *ne_guess, gas_state *gs, do_cool_data *coolpars);
double CoolingRateFromU(double u, double rho, double *ne_guess);
double CoolingLookup(double logT, double logrho, double *ne_guess);
double convert_u_to_temp(double u, double rho, double *ne_guess);
void find_abundances_and_rates(double logT, double rho, double *ne);
double CoolingRate(double logT, double rho, double ne, int mode);
void MakeRadTab();
void SetZeroIonization();

// Helper function for program termination
void endrun(int errorcode)
{
  printf("Error code %d: terminating program\n", errorcode);
  exit(errorcode);
}

void coolsfr::InitCool()
{
  // Set default values for cooling table parameters
  rhomin = -6.0;
  rhomax = 3.0;
  Tmin = 2.0;
  Tmax = 9.0;

  // Read TREECOOL file and initialize ionization parameters
  ReadIonizeParams("data/TREECOOL");
  
  // Initialize ionization parameters
  IonizeParams();
  
  // Create cooling tables
  MakeCoolingTable();
  
  // Set table spacing
  dlogT = (Tmax - Tmin) / (TEMP_TABLE_SIZE - 1);
  dlogRho = (RhoMax - RhoMin) / (RHO_TABLE_SIZE - 1);
  
  // Initialize MakeRadTab
  MakeRadTab();
  
  if(ThisTask == 0)
    printf("COOLING: Cooling tables initialized successfully.\n");
}

void coolsfr::ReadIonizeParams(const char* fname)
{
    FILE* fdcool = std::fopen(fname, "r");

    if (!fdcool)
    {
        printf("Cannot read ionization table in file `%s`\n", fname);
        endrun(456);
    }

    for (int i = 0; i < TABLESIZE; ++i)
        gH0[i] = 0.0;

    for (int i = 0; i < TABLESIZE; ++i)
    {
        int ret = std::fscanf(fdcool, "%g %g %g %g %g %g %g",
                              &inlogz[i], &gH0[i], &gHe[i], &gHep[i],
                              &eH0[i], &eHe[i], &eHep[i]);

        if (ret == EOF)
            break;
    }

    std::fclose(fdcool);

    // Count valid entries in table
    nheattab = 0;
    for (int i = 0; i < TABLESIZE; ++i)
    {
        if (gH0[i] != 0.0)
            ++nheattab;
        else
            break;
    }

    if (ThisTask == 0)
        printf("\n\nCOOLING: Read ionization table with %d entries in file `%s`.\n\n", nheattab, fname);
}

void coolsfr::MakeCoolingTable()
{
    const int N = NCOOLTAB;
    double T, Tfact;

    yhelium = (1.0 - XH) / (4.0 * XH);
    mhboltz = PROTONMASS / BOLTZMANN;

    if (All.MinGasTemp > 0.0)
        Tmin = log10(All.MinGasTemp);
    else
        Tmin = 1.0;

    this->deltaT = (this->Tmax - this->Tmin) / N;

    this->ethmin = std::pow(10.0, this->Tmin) * (1.0 + yhelium) /
                   ((1.0 + 4.0 * yhelium) * mhboltz * GAMMA_MINUS1);

    for (int i = 0; i <= N; ++i)
    {
        BetaH0[i] = BetaHep[i] = Betaff[i] = 0;
        AlphaHp[i] = AlphaHep[i] = AlphaHepp[i] = Alphad[i] = 0;
        GammaeH0[i] = GammaeHe0[i] = GammaeHep[i] = 0;

        T = std::pow(10.0, this->Tmin + this->deltaT * i);
        Tfact = 1.0 / (1.0 + std::sqrt(T / 1.0e5));

        if (118348.0 / T < 70.0)
            BetaH0[i] = 7.5e-19 * std::exp(-118348.0 / T) * Tfact;

        if (473638.0 / T < 70.0)
            BetaHep[i] = 5.54e-17 * std::pow(T, -0.397) * std::exp(-473638.0 / T) * Tfact;

        Betaff[i] = 1.43e-27 * std::sqrt(T) * (1.1 + 0.34 * std::exp(-std::pow((5.5 - log10(T)), 2) / 3.0));

        AlphaHp[i] = 8.4e-11 * std::pow(T / 1000.0, -0.2) / (1.0 + std::pow(T / 1.0e6, 0.7)) / std::sqrt(T);
        AlphaHep[i] = 1.5e-10 * std::pow(T, -0.6353);
        AlphaHepp[i] = 4.0 * AlphaHp[i];

        if (470000.0 / T < 70.0)
            Alphad[i] = 1.9e-3 * std::pow(T, -1.5) * std::exp(-470000.0 / T) *
                        (1.0 + 0.3 * std::exp(-94000.0 / T));

        if (157809.1 / T < 70.0)
            GammaeH0[i] = 5.85e-11 * std::sqrt(T) * std::exp(-157809.1 / T) * Tfact;

        if (285335.4 / T < 70.0)
            GammaeHe0[i] = 2.38e-11 * std::sqrt(T) * std::exp(-285335.4 / T) * Tfact;

        if (631515.0 / T < 70.0)
            GammaeHep[i] = 5.68e-12 * std::sqrt(T) * std::exp(-631515.0 / T) * Tfact;
    }
}

void coolsfr::IonizeParams()
{
    // Get UV background from the TREECOOL table
    IonizeParamsTable();

    // Initialize the cooling tables with current ionization parameters
    MakeRadTab();
}

void coolsfr::IonizeParamsTable()
{
    if (!All.ComovingIntegrationOn)
    {
        gJHe0 = gJHep = gJH0 = 0.0;
        epsHe0 = epsHep = epsH0 = 0.0;
        J_UV = 0.0;
        return;
    }

    double redshift = 1.0 / All.Time - 1.0;
    float logz = std::log10(redshift + 1.0);

    int ilow = 0;
    for (int i = 0; i < nheattab; ++i)
    {
        if (inlogz[i] < logz)
            ilow = i;
        else
            break;
    }

    float dzlow = logz - inlogz[ilow];
    float dzhi  = inlogz[ilow + 1] - logz;

    if (logz > inlogz[nheattab - 1] || gH0[ilow] == 0.0 || gH0[ilow + 1] == 0.0 || nheattab == 0)
    {
        gJHe0 = gJHep = gJH0 = 0.0;
        epsHe0 = epsHep = epsH0 = 0.0;
        J_UV = 0.0;
        return;
    }
    else
    {
        J_UV = 1.0e-21; // Arbitrary nonzero UV field strength
    }

    // Interpolate in log-space between ilow and ilow+1
    auto interpolate = [&](float y1, float y2) {
        return JAMPL * std::pow(10.0, (dzhi * std::log10(y1) + dzlow * std::log10(y2)) / (dzlow + dzhi));
    };

    gJH0   = interpolate(gH0[ilow],  gH0[ilow + 1]);
    gJHe0  = interpolate(gHe[ilow],  gHe[ilow + 1]);
    gJHep  = interpolate(gHep[ilow], gHep[ilow + 1]);
    epsH0  = interpolate(eH0[ilow],  eH0[ilow + 1]);
    epsHe0 = interpolate(eHe[ilow],  eHe[ilow + 1]);
    epsHep = interpolate(eHep[ilow], eHep[ilow + 1]);
}

// Set ionization rates to zero (used when UV background is disabled)
void SetZeroIonization()
{
    gJHe0 = gJHep = gJH0 = 0.0;
    epsHe0 = epsHep = epsH0 = 0.0;
    J_UV = 0.0;
}

// Find abundance ratios
void find_abundances_and_rates(double logT, double rho, double *ne)
{
    double T = std::pow(10.0, logT);
    double T_eV = T * BOLTZMANN / ELECTRONVOLT;
    
    // Simple electron density approximation for fully ionized gas
    // This is a simplified version - for full accuracy you'd need 
    // to solve the ionization balance equations
    *ne = 1.0 + 2.0 * yhelium;  // Fully ionized H and He
    
    // If you have tabulated values, you could interpolate here
}

// Get cooling rate for a specific temperature and density
double CoolingRate(double logT, double rho, double ne, int mode)
{
    double T = std::pow(10.0, logT);
    double Lambda = 0.0;
    
    // Calculate cooling due to various processes
    // This is a simplified version
    double LambdaExcH0, LambdaExcHep, LambdaIonH0, LambdaIonHe0, LambdaIonHep;
    double LambdaRecHp, LambdaRecHep, LambdaRecHepp, LambdaRecHepd;
    double LambdaFF, Heat;
    
    // Lookup cooling rate from table or calculate directly
    // For simplicity, returning a basic cooling function
    double n_H = rho * XH / PROTONMASS;
    
    // A simple cooling function approximation
    if (T > 1.0e4) {
        // Bremsstrahlung dominates at high temperatures
        Lambda = 1.42e-27 * std::sqrt(T) * ne * (n_H + 4.0 * n_H * yhelium);
    } else {
        // Recombination cooling
        Lambda = 1.0e-23 * T * ne * n_H;
    }
    
    // Include photoheating
    double Heat = (n_H * epsH0 + n_H * yhelium * epsHe0) / rho;
    
    // Net cooling
    return Heat - Lambda;
}

void MakeRadTab()
{
    int irho, itemp;
    double logT, rho, dT, drho;
    double ne;
    double redshift;

    redshift = 1.0 / All.Time - 1.0;

    // Check if the background or redshift changed significantly
    if (std::fabs(gJH0_old - gJH0) / (gJH0 + SMALLNUM) < 0.01 &&
        std::fabs(redshift - redshift_old) < 1.0)
        return;

    drho = (rhomax - rhomin) / NRHOTAB;
    dT = (Tmax - Tmin) / NTEMPTAB;
    drhoinv = 1.0 / drho;
    dTinv = 1.0 / dT;

    // Fill the interpolation arrays
    for (irho = 0; irho < NRHOTAB; ++irho)
        rhoarray[irho] = drho * irho + rhomin;

    for (itemp = 0; itemp < NTEMPTAB; ++itemp)
        Tarray[itemp] = dT * itemp + Tmin;

    // Populate the cooling and electron density tables
    nH0 = nHe0 = 1.0;
    nHep = nHepp = 0.0;

    for (irho = 0; irho < NRHOTAB; ++irho)
    {
        rho = std::pow(10.0, irho * drho + rhomin);

        for (itemp = 0; itemp < NTEMPTAB; ++itemp)
        {
            logT = itemp * dT + Tmin;
            ne = 1.0; // Initial guess
            find_abundances_and_rates(logT, rho, &ne);
            neTable[irho][itemp] = ne;
            coolTable[irho][itemp] = CoolingRate(logT, rho, ne, EQUILIBRIUM_IONIZED);
        }
    }

    if (ThisTask == 0 && npcool > 0)
        printf("COOLING: average cooling iterations = %g (%d particles cooled)\n", 1.0 * itercool / npcool, npcool);

    if (ThisTask == 0)
        printf("COOLING: Made New Cooling Table at z = %g (dJH0 = %g)\n", redshift, std::fabs(gJH0_old - gJH0) / (gJH0 + SMALLNUM));

    npcool = itercool = 0;
    gJH0_old = gJH0;
    redshift_old = redshift;
}

double CoolingLookup(double logT, double logrho, double *ne_guess)
{
  int irho = (logrho - RhoMin) / dlogRho;
  int itemp = (logT - Tmin) / dlogT;

  if (irho < 0) irho = 0;
  if (irho >= RHO_TABLE_SIZE - 1) irho = RHO_TABLE_SIZE - 2;
  if (itemp < 0) itemp = 0;
  if (itemp >= TEMP_TABLE_SIZE - 1) itemp = TEMP_TABLE_SIZE - 2;

  double x1 = (logrho - (RhoMin + irho * dlogRho)) / dlogRho;
  double x2 = (logT - (Tmin + itemp * dlogT)) / dlogT;

  double dudt =
    (1 - x1) * (1 - x2) * coolTable[irho][itemp] +
    x1 * (1 - x2) * coolTable[irho + 1][itemp] +
    (1 - x1) * x2 * coolTable[irho][itemp + 1] +
    x1 * x2 * coolTable[irho + 1][itemp + 1];

  *ne_guess = (1 - x1) * (1 - x2) * neTable[irho][itemp] +
              x1 * (1 - x2) * neTable[irho + 1][itemp] +
              (1 - x1) * x2 * neTable[irho][itemp + 1] +
              x1 * x2 * neTable[irho + 1][itemp + 1];

  return dudt;
}

double convert_u_to_temp(double u, double rho, double *ne_guess)
{
    double temp;
    double mu = (1.0 + 4.0 * yhelium) / (1.0 + yhelium + *ne_guess);
    
    temp = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS * mu;
    
    return temp;
}

static double CoolingRateFromU(double u, double rho, double *ne_guess)
{
  double temp = convert_u_to_temp(u, rho, ne_guess);
  double logT = log10(temp);
  double logrho = log10(rho);
  return CoolingLookup(logT, logrho, ne_guess);
}

double coolsfr::DoCooling(double u_old, double rho, double dt, double *ne_guess, gas_state *gs, do_cool_data *coolpars)
{
  double u, LambdaNet;
  double du, dt_cool;
  int iter = 0;
  double m_old, m, m_lower, m_upper, dm;

  DoCool_u_old_input = u_old;
  DoCool_rho_input = rho;
  DoCool_dt_input = dt;
  DoCool_ne_guess_input = *ne_guess;

  rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;
  u_old *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;

  double nHcgs = XH * rho / PROTONMASS;
  double ratefact = nHcgs * nHcgs / rho;

  u = u_old;
  m = m_old = 1.0;
  m_lower = 1.0 / 1.1;
  m_upper = 1.0 * 1.1;

  LambdaNet = CoolingRateFromU(u, rho, ne_guess);

  if(LambdaNet >= 0)
    return u_old;

  if(m - m_old - m_old * ratefact * LambdaNet * dt > 0)
  {
    m_lower /= sqrt(1.1);
    m_upper *= sqrt(1.1);
    while(m_lower - m_old - m_lower * m_lower / m_old * ratefact * CoolingRateFromU(u, rho * m_lower / m_old, ne_guess) * dt > 0)
    {
      m_upper /= 1.1;
      m_lower /= 1.1;
    }
  }

  do
  {
    m = 0.5 * (m_lower + m_upper);
    LambdaNet = CoolingRateFromU(u, rho * m / m_old, ne_guess);

    if(m - m_old - m * m / m_old * ratefact * LambdaNet * dt > 0)
      m_upper = m;
    else
      m_lower = m;

    dm = m_upper - m_lower;
    iter++;

    if(iter >= (MAXITER - 10))
      printf("m= %g\n", m);
  }
  while(fabs(dm / m) > 1.0e-6 && iter < MAXITER);

  if(iter >= MAXITER)
  {
    printf("failed to converge in DoCooling()\n");
    printf("DoCool_u_old_input=%g\nDoCool_rho_input= %g\nDoCool_dt_input= %g\nDoCool_ne_guess_input= %g\n",
      DoCool_u_old_input, DoCool_rho_input, DoCool_dt_input, DoCool_ne_guess_input);
    printf("m_old= %g\n", m_old);
    endrun(11);
  }

  u = u_old * m / m_old;
  u *= All.UnitDensity_in_cgs / All.UnitPressure_in_cgs;
  return u;
}

// Main routine called to cool gas particles
void coolsfr::cooling_only(simparticles *Sp)
{
  TIMER_START(CPU_COOLING);
  
  // Update cosmological factors and ionization parameters
  All.set_cosmo_factors_for_current_time();
  IonizeParams();
  
  // Setup cooling structures
  gas_state gs = GasState;
  do_cool_data DoCool = DoCoolData;
  
  // Process all active SPH particles
  for(int i = 0; i < Sp->TimeBinsHydro.NActiveParticles; i++)
  {
    int target = Sp->TimeBinsHydro.ActiveParticleList[i];
    if(Sp->P[target].getType() == 0)
    {
      if(Sp->P[target].getMass() == 0 && Sp->P[target].ID.get() == 0)
        continue; /* skip particles that have been swallowed or eliminated */
      
      // Apply cooling to this particle
      cool_sph_particle(Sp, target, &gs, &DoCool);
    }
  }
  
  TIMER_STOP(CPU_COOLING);
}

// Apply cooling to a single SPH particle
void coolsfr::cool_sph_particle(simparticles *Sp, int i, gas_state *gs, do_cool_data *DoCool)
{
  // Get particle properties
  double dt = (Sp->P[i].getTimeBinHydro() ? (((integertime)1) << Sp->P[i].getTimeBinHydro()) : 0) * All.Timebase_interval;
  double ne = Sp->SphP[i].Ne;
  double unew;
  
  // Apply cooling
  unew = DoCooling(Sp->SphP[i].Utherm, Sp->get_dens_around_particle(i), dt, &ne, gs, DoCool);
  
  // Update particle properties
  Sp->SphP[i].Ne = ne;
  Sp->SphP[i].Utherm = unew;
  
  // Also update entropy if needed
  Sp->set_entropy_from_utherm(unew, i);
}

#endif /* COOLING */