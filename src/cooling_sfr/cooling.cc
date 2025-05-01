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
#include "../data/constants.h"
#include "../logs/logs.h"
#include "../logs/timer.h"
#include "../system/system.h"
#include "../time_integration/timestep.h"

// Cooling table definitions
#define JAMPL 1.0
#define TABLESIZE 500        // Maximum rows in TREECOOL file
#define NCOOLTAB 2000
#define XH 0.76
#define COOLLIM 0.01
#define COOL_TOL 0.002
#define MIN_DTIONFRAC 0.01   // Minimum ionization substep size
#define SMALLNUM 1.0e-60
#define MAXITER 1000
#define ELECTRONVOLT 1.60217733e-12

// TREECOOL ionization background table
static float inlogz[TABLESIZE];   // log10(z+1)
static float gH0[TABLESIZE];      // photoionization rate for H0
static float gHe[TABLESIZE];      // photoionization rate for He0
static float gHep[TABLESIZE];     // photoionization rate for He+
static float eH0[TABLESIZE];      // photoheating rate for H0
static float eHe[TABLESIZE];      // photoheating rate for He0
static float eHep[TABLESIZE];     // photoheating rate for He+
static int nheattab = 0;          // actual length of the table

// Global ionization parameters
static double J_UV = 0, gJH0 = 0, gJHep = 0, gJHe0 = 0, epsH0 = 0, epsHep = 0, epsHe0 = 0;
static double redshift_old = -100., gJH0_old = -100.;

// Cooling rate tables
static double BetaH0[NCOOLTAB], BetaHep[NCOOLTAB], Betaff[NCOOLTAB];
static double AlphaHp[NCOOLTAB], AlphaHep[NCOOLTAB], AlphaHepp[NCOOLTAB], Alphad[NCOOLTAB];
static double GammaeH0[NCOOLTAB], GammaeHe0[NCOOLTAB], GammaeHep[NCOOLTAB];

// Helper variables
static double yhelium;
static double mhboltz;

void coolsfr::InitCool()
{
  // Initialize physical constants
  yhelium = (1.0 - XH) / (4.0 * XH);
  mhboltz = PROTONMASS / BOLTZMANN;

  // Initialize cooling tables
  MakeCoolingTable();
  
  // Read in ionization parameters from TREECOOL file
  if(ThisTask == 0)
    printf("COOLING: Reading ionization table from file '%s'\n", All.TreecoolFile);
  
  ReadIonizeParams(All.TreecoolFile);
  
  // Set up ionization parameters
  IonizeParams();
  
  if(ThisTask == 0)
    printf("COOLING: Cooling tables initialized successfully.\n");
}

void coolsfr::ReadIonizeParams(char *fname)
{
    FILE* fdcool = fopen(fname, "r");

    if (!fdcool)
    {
        printf("COOLING: Cannot read ionization table in file '%s'\n", fname);
        Terminate("Failed to open TREECOOL file\n");
    }

    for (int i = 0; i < TABLESIZE; ++i)
        gH0[i] = 0.0;

    for (int i = 0; i < TABLESIZE; ++i)
    {
        int ret = fscanf(fdcool, "%g %g %g %g %g %g %g",
                          &inlogz[i], &gH0[i], &gHe[i], &gHep[i],
                          &eH0[i], &eHe[i], &eHep[i]);

        if (ret == EOF)
            break;
    }

    fclose(fdcool);

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
        printf("COOLING: Read ionization table with %d entries in file '%s'.\n", nheattab, fname);
}

void coolsfr::MakeCoolingTable()
{
    // Set up the temperature range based on MinGasTemp
    if (All.MinGasTemp > 0.0)
        deltaT = (log10(All.MaxGasTemp) - log10(All.MinGasTemp)) / NCOOLTAB;
    else
        deltaT = (log10(1.0e9) - log10(100.0)) / NCOOLTAB;

    // Calculate ethmin based on MinGasTemp
    ethmin = All.MinGasTemp * (1.0 + yhelium) / ((1.0 + 4.0 * yhelium) * mhboltz * GAMMA_MINUS1);

    for (int i = 0; i <= NCOOLTAB; ++i)
    {
        BetaH0[i] = BetaHep[i] = Betaff[i] = 0;
        AlphaHp[i] = AlphaHep[i] = AlphaHepp[i] = Alphad[i] = 0;
        GammaeH0[i] = GammaeHe0[i] = GammaeHep[i] = 0;

        double T = pow(10.0, log10(All.MinGasTemp) + deltaT * i);
        double Tfact = 1.0 / (1.0 + sqrt(T / 1.0e5));

        if (118348.0 / T < 70.0)
            BetaH0[i] = 7.5e-19 * exp(-118348.0 / T) * Tfact;

        if (473638.0 / T < 70.0)
            BetaHep[i] = 5.54e-17 * pow(T, -0.397) * exp(-473638.0 / T) * Tfact;

        Betaff[i] = 1.43e-27 * sqrt(T) * (1.1 + 0.34 * exp(-pow((5.5 - log10(T)), 2) / 3.0));

        AlphaHp[i] = 8.4e-11 * pow(T / 1000.0, -0.2) / (1.0 + pow(T / 1.0e6, 0.7)) / sqrt(T);
        AlphaHep[i] = 1.5e-10 * pow(T, -0.6353);
        AlphaHepp[i] = 4.0 * AlphaHp[i];

        if (470000.0 / T < 70.0)
            Alphad[i] = 1.9e-3 * pow(T, -1.5) * exp(-470000.0 / T) *
                      (1.0 + 0.3 * exp(-94000.0 / T));

        if (157809.1 / T < 70.0)
            GammaeH0[i] = 5.85e-11 * sqrt(T) * exp(-157809.1 / T) * Tfact;

        if (285335.4 / T < 70.0)
            GammaeHe0[i] = 2.38e-11 * sqrt(T) * exp(-285335.4 / T) * Tfact;

        if (631515.0 / T < 70.0)
            GammaeHep[i] = 5.68e-12 * sqrt(T) * exp(-631515.0 / T) * Tfact;
    }
}

void coolsfr::IonizeParams()
{
    if (!All.ComovingIntegrationOn)
    {
        SetZeroIonization();
        return;
    }

    // Get UV background from the TREECOOL table
    IonizeParamsUVB();
}

void coolsfr::IonizeParamsUVB()
{
    if (!All.ComovingIntegrationOn)
    {
        gJHe0 = gJHep = gJH0 = 0.0;
        epsHe0 = epsHep = epsH0 = 0.0;
        J_UV = 0.0;
        return;
    }

    double redshift = 1.0 / All.Time - 1.0;
    float logz = log10(redshift + 1.0);

    // If nheattab is too small, disable UV background
    if (nheattab < 2) 
    {
        J_UV = gJHe0 = gJHep = gJH0 = epsHe0 = epsHep = epsH0 = 0.0;
        return;
    }

    // Find position in TREECOOL table
    int ilow = 0;
    for (int i = 0; i < nheattab - 1; ++i)
    {
        if (inlogz[i] < logz)
            ilow = i;
        else
            break;
    }

    // If redshift is higher than in table, disable UV background
    if (logz > inlogz[nheattab - 1] || gH0[ilow] == 0.0 || gH0[ilow + 1] == 0.0)
    {
        J_UV = gJHe0 = gJHep = gJH0 = epsHe0 = epsHep = epsH0 = 0.0;
        return;
    }
    else
    {
        J_UV = 1.0e-21; // Arbitrary nonzero UV field strength
    }

    // Interpolate in log-space between ilow and ilow+1
    float dzlow = logz - inlogz[ilow];
    float dzhi  = inlogz[ilow + 1] - logz;
    
    // Helper function for log-space interpolation
    auto interpolate = [&](float y1, float y2) {
        return JAMPL * pow(10.0, (dzhi * log10(y1) + dzlow * log10(y2)) / (dzlow + dzhi));
    };

    gJH0   = interpolate(gH0[ilow],  gH0[ilow + 1]);
    gJHe0  = interpolate(gHe[ilow],  gHe[ilow + 1]);
    gJHep  = interpolate(gHep[ilow], gHep[ilow + 1]);
    epsH0  = interpolate(eH0[ilow],  eH0[ilow + 1]);
    epsHe0 = interpolate(eHe[ilow],  eHe[ilow + 1]);
    epsHep = interpolate(eHep[ilow], eHep[ilow + 1]);
}

void coolsfr::SetZeroIonization()
{
    J_UV = gJHe0 = gJHep = gJH0 = epsHe0 = epsHep = epsH0 = 0.0;
}

// Look up cooling rate for a specific temperature
double coolsfr::CoolingRateGetU(double logT, double rho, double *ne_guess, double *nH0, double *nHe0, double *nHep)
{
    double T = pow(10.0, logT);
    double T_cmb = 2.7255 / All.Time; // CMB temperature
    
    // Find position in cooling table
    int tindex = (logT - log10(All.MinGasTemp)) / deltaT;
    if (tindex < 0) tindex = 0;
    if (tindex >= NCOOLTAB) tindex = NCOOLTAB - 1;
    
    // Simple electron density estimation
    if (*ne_guess <= 0.0) *ne_guess = 1.0;
    
    // Simplified cooling rate calculation
    double Lambda;
    
    // Recombination cooling
    Lambda = AlphaHp[tindex] * (*nH0) * (*ne_guess) * 2.18e-11;
    Lambda += AlphaHep[tindex] * (*nHep) * (*ne_guess) * 1.103e-10;
    Lambda += AlphaHepp[tindex] * (*nHe0) * (*ne_guess) * 3.94e-10;
    
    // Collisional ionization cooling
    Lambda += GammaeH0[tindex] * (*nH0) * (*ne_guess) * 1.361e-11;
    Lambda += GammaeHe0[tindex] * (*nHe0) * (*ne_guess) * 2.469e-11;
    Lambda += GammaeHep[tindex] * (*nHep) * (*ne_guess) * 5.526e-11;
    
    // Collisional excitation cooling
    Lambda += BetaH0[tindex] * (*nH0) * (*ne_guess);
    Lambda += BetaHep[tindex] * (*nHep) * (*ne_guess);
    
    // Free-free emission (Bremsstrahlung)
    Lambda += Betaff[tindex] * (*ne_guess) * ((*nH0) + (*nHep) + 4.0 * (*nHe0));
    
    // Compton cooling/heating
    double redshift = 1.0 / All.Time - 1.0;
    double LambdaCmptn = 5.65e-36 * (*ne_guess) * (T - T_cmb) * pow(1.0 + redshift, 4) / rho;
    Lambda += LambdaCmptn;
    
    // Add photoheating from UV background
    double Heat = 0.0;
    if (J_UV != 0.0) {
        Heat = (*nH0) * epsH0 + (*nHe0) * epsHe0 + (*nHep) * epsHep;
        Heat /= rho;
    }
    
    return Heat - Lambda;
}

// Calculate cooling rate from internal energy
double coolsfr::CoolingRateFromU(double u, double rho, double *ne_guess, gas_state *gs, const do_cool_data *DoCool)
{
    // Convert internal energy to temperature
    double temp = convert_u_to_temp(u, rho, ne_guess, gs, DoCool);
    double logT = log10(temp);
    
    // Estimate abundances (simplified)
    double nH0 = XH * rho / PROTONMASS * 0.76;    // Mostly neutral at typical ISM temperatures
    double nHe0 = nH0 * yhelium * 0.1;            // Mostly neutral helium
    double nHep = nH0 * yhelium * 0.9;            // Some singly-ionized helium
    
    // Get cooling rate
    return CoolingRateGetU(logT, rho, ne_guess, &nH0, &nHe0, &nHep);
}

// Convert internal energy to temperature 
double coolsfr::convert_u_to_temp(double u, double rho, double *ne_guess, gas_state *gs, const do_cool_data *DoCool)
{
    double mu = (1.0 + 4.0 * yhelium) / (1.0 + yhelium + *ne_guess);
    double temp = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS * mu;
    return temp;
}

double coolsfr::DoCooling(double u_old, double rho, double dt, double *ne_guess, gas_state *gs, do_cool_data *DoCool)
{
    if(dt == 0)
        return u_old;
        
    // Save input values
    DoCool->u_old_input = u_old;
    DoCool->rho_input = rho;
    DoCool->dt_input = dt;
    DoCool->ne_guess_input = *ne_guess;

    // Convert to physical cgs units
    rho *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;
    u_old *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;

    double nHcgs = XH * rho / PROTONMASS;
    double ratefact = nHcgs * nHcgs / rho;

    double u = u_old;
    double LambdaNet = CoolingRateFromU(u, rho, ne_guess, gs, DoCool);

    // If net heating, just return old internal energy
    if(LambdaNet >= 0)
        return u_old;

    // Use semi-implicit scheme for cooling
    double u_new = u_old / (1.0 - dt * ratefact * LambdaNet / u_old);
    
    // Limit cooling to reasonable values
    if(u_new < 0.1 * u_old)
        u_new = 0.1 * u_old;
        
    // Convert back to code units and return
    u_new *= All.UnitDensity_in_cgs / All.UnitPressure_in_cgs;
    return u_new;
}

void coolsfr::cooling_only(simparticles *Sp)
{
    TIMER_START(CPU_COOLING_SFR);
    
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
    
    TIMER_STOP(CPU_COOLING_SFR);
}

void coolsfr::cool_sph_particle(simparticles *Sp, int i, gas_state *gs, do_cool_data *DoCool)
{
    // Get particle properties
    double dt = (Sp->P[i].getTimeBinHydro() ? (((integertime)1) << Sp->P[i].getTimeBinHydro()) : 0) * All.Timebase_interval;
    double ne = Sp->SphP[i].Ne;
    double rho = Sp->get_density(i);
    double u_old = Sp->get_utherm(i);
    
    // Apply cooling
    double unew = DoCooling(u_old, rho, dt, &ne, gs, DoCool);
    
    // Update particle properties
    Sp->SphP[i].Ne = ne;
    Sp->update_internal_energy(i, unew);
}

#endif /* COOLING */