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
#include "../logs/logs.h"
#include "../logs/timer.h"
#include "../system/system.h"
#include "../time_integration/timestep.h"
#include "../data/constants.h"

// Define missing constant
#define VERY_LARGE_NUMBER 1.0e30

// TREECOOL table variables
#define JAMPL 1.0
#define TABLESIZE 500
#define NCOOLTAB 2000

static float inlogz[TABLESIZE];   // log10(z+1)
static float gH0[TABLESIZE];      // photoionization rate for H0
static float gHe[TABLESIZE];      // photoionization rate for He0
static float gHep[TABLESIZE];     // photoionization rate for He+
static float eH0[TABLESIZE];      // photoheating rate for H0
static float eHe[TABLESIZE];      // photoheating rate for He0
static float eHep[TABLESIZE];     // photoheating rate for He+
static int nheattab = 0;          // actual length of the table

// Cooling rate tables
static double BetaH0[NCOOLTAB], BetaHep[NCOOLTAB], Betaff[NCOOLTAB];
static double AlphaHp[NCOOLTAB], AlphaHep[NCOOLTAB], AlphaHepp[NCOOLTAB], Alphad[NCOOLTAB];
static double GammaeH0[NCOOLTAB], GammaeHe0[NCOOLTAB], GammaeHep[NCOOLTAB];

// Global ionization parameters
static double J_UV = 0, gJH0 = 0, gJHep = 0, gJHe0 = 0, epsH0 = 0, epsHep = 0, epsHe0 = 0;

void coolsfr::InitCool()
{
  // Initialize physical constants
  MakeCoolingTable();
  
  // Read in ionization parameters from TREECOOL file
  if(ThisTask == 0)
    mpi_printf("COOLING: Reading ionization table from file '%s'\n", All.TreecoolFile);
  
  ReadIonizeParams(All.TreecoolFile);
  
  // Set up ionization parameters based on current time
  All.Time = All.TimeBegin;
  All.set_cosmo_factors_for_current_time();
  
  IonizeParams();
  
  if(ThisTask == 0)
    mpi_printf("COOLING: Cooling tables initialized successfully.\n");
}

void coolsfr::ReadIonizeParams(char *fname)
{
    FILE* fdcool = fopen(fname, "r");

    if (!fdcool)
    {
        mpi_printf("COOLING: Cannot read ionization table in file '%s'\n", fname);
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
        mpi_printf("COOLING: Read ionization table with %d entries in file '%s'.\n", nheattab, fname);

    // Ignore zeros at end of treecool file
    for(int i = 0; i < nheattab; ++i) {
        if(gH0[i] == 0.0) {
            nheattab = i;
            break;
        }
    }

    if(nheattab < 1)
        Terminate("The length of the cooling table has to have at least one entry");
}

void coolsfr::MakeCoolingTable()
{
    const int N = NCOOLTAB;
    double T, Tfact;

    double yhelium = (1.0 - HYDROGEN_MASSFRAC) / (4.0 * HYDROGEN_MASSFRAC);
    double mhboltz = PROTONMASS / BOLTZMANN;

    // Set up the temperature range
    double Tmin = 1.0;
    double Tmax = 9.0;

    if (All.MinGasTemp > 0.0)
        Tmin = log10(All.MinGasTemp);

    this->deltaT = (Tmax - Tmin) / N;

    // Note: Using All.MinGasTemp instead of ethmin
    // The ethmin variable is not used elsewhere in this implementation
    double min_temp = pow(10.0, Tmin) * (1.0 + yhelium) / ((1.0 + 4.0 * yhelium) * mhboltz * GAMMA_MINUS1);

    for (int i = 0; i <= N; ++i)
    {
        BetaH0[i] = BetaHep[i] = Betaff[i] = 0;
        AlphaHp[i] = AlphaHep[i] = AlphaHepp[i] = Alphad[i] = 0;
        GammaeH0[i] = GammaeHe0[i] = GammaeHep[i] = 0;

        T = pow(10.0, Tmin + this->deltaT * i);
        Tfact = 1.0 / (1.0 + sqrt(T / 1.0e5));

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
        SetZeroIonization();
        return;
    }

    double redshift = 1.0 / All.Time - 1.0;
    
    if (nheattab == 1)
    {
        // Treat the one value given as constant with redshift
        J_UV = 1;
        gJH0 = gH0[0];
        gJHe0 = gHe[0];
        gJHep = gHep[0];
        epsH0 = eH0[0];
        epsHe0 = eHe[0];
        epsHep = eHep[0];
        return;
    }

    float logz = log10(redshift + 1.0);
    int ilow = 0;
    
    for (int i = 0; i < nheattab; i++)
    {
        if (inlogz[i] < logz)
            ilow = i;
        else
            break;
    }

    if (logz > inlogz[nheattab - 1] || ilow >= nheattab - 1)
    {
        // Beyond table range, use the last value
        J_UV = 1;
        gJH0 = gH0[nheattab - 1];
        gJHe0 = gHe[nheattab - 1];
        gJHep = gHep[nheattab - 1];
        epsH0 = eH0[nheattab - 1];
        epsHe0 = eHe[nheattab - 1];
        epsHep = eHep[nheattab - 1];
        return;
    }

    float dzlow = logz - inlogz[ilow];
    float dzhi = inlogz[ilow + 1] - logz;

    if (gH0[ilow] == 0 || gH0[ilow + 1] == 0)
    {
        SetZeroIonization();
        return;
    }

    J_UV = 1;
    
    // Interpolate in log space
    gJH0 = pow(10.0, (dzhi * log10(gH0[ilow]) + dzlow * log10(gH0[ilow + 1])) / (dzlow + dzhi));
    gJHe0 = pow(10.0, (dzhi * log10(gHe[ilow]) + dzlow * log10(gHe[ilow + 1])) / (dzlow + dzhi));
    gJHep = pow(10.0, (dzhi * log10(gHep[ilow]) + dzlow * log10(gHep[ilow + 1])) / (dzlow + dzhi));
    epsH0 = pow(10.0, (dzhi * log10(eH0[ilow]) + dzlow * log10(eH0[ilow + 1])) / (dzlow + dzhi));
    epsHe0 = pow(10.0, (dzhi * log10(eHe[ilow]) + dzlow * log10(eHe[ilow + 1])) / (dzlow + dzhi));
    epsHep = pow(10.0, (dzhi * log10(eHep[ilow]) + dzlow * log10(eHep[ilow + 1])) / (dzlow + dzhi));
}

void coolsfr::SetZeroIonization()
{
    J_UV = gJHe0 = gJHep = gJH0 = epsHe0 = epsHep = epsH0 = 0.0;
}

double coolsfr::GetCoolingTime(double u_old, double rho, double *ne_guess, gas_state *gs, do_cool_data *DoCool)
{
    DoCool->u_old_input = u_old;
    DoCool->rho_input = rho;
    DoCool->ne_guess_input = *ne_guess;

    if(!gsl_finite(u_old))
        Terminate("invalid input: u_old=%g\n", u_old);

    if(u_old < 0 || rho < 0)
        return 0;

    double u = u_old;
    gs->nHcgs = rho * HYDROGEN_MASSFRAC / PROTONMASS;
    double ratefact = gs->nHcgs * gs->nHcgs / rho;

    double LambdaNet = CoolingRateFromU(u, rho, ne_guess, gs, DoCool);

    if(LambdaNet >= 0)
        return 1.0e30;  // Very large number instead of VERY_LARGE_NUMBER

    double coolingtime = u_old / (-ratefact * LambdaNet);

    return coolingtime;
}

double coolsfr::convert_u_to_temp(double u, double rho, double *ne_guess, gas_state *gs, const do_cool_data *DoCool)
{
    double mu = (1.0 + 4.0 * gs->yhelium) / (1.0 + gs->yhelium + *ne_guess);
    double temp = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS * mu;
    
    return temp;
}

void coolsfr::find_abundances_and_rates(double logT, double rho, double *ne_guess, gas_state *gs, const do_cool_data *DoCool)
{
    double T = pow(10.0, logT);
    
    if(!gsl_finite(logT))
        Terminate("invalid input: logT=%g\n", logT);

    gs->nH0 = gs->nHe0 = 1.0;
    gs->nHp = gs->nHep = gs->nHepp = 0.0;
    gs->ne = *ne_guess;

    if(T >= 1.0e9)
    {
        // Fully ionized
        gs->nH0 = 0.0;
        gs->nHe0 = 0.0;
        gs->nHep = 0.0;
        gs->nHp = 1.0;
        gs->nHepp = gs->yhelium;
        gs->ne = gs->nHp + 2.0 * gs->nHepp;
        *ne_guess = gs->ne;
        return;
    }

    int j = (logT - log10(100.0)) / this->deltaT;
    if(j < 0) j = 0;
    if(j >= NCOOLTAB) j = NCOOLTAB - 1;

    double fhi = (logT - (log10(100.0) + j * this->deltaT)) / this->deltaT;
    double flow = 1.0 - fhi;

    // Initialize with previous values
    double neold = gs->ne;
    gs->necgs = gs->ne * gs->nHcgs;

    // Iteratively solve for ionization equilibrium
    int niter = 0;
    do
    {
        gs->aHp = flow * AlphaHp[j] + fhi * AlphaHp[j + 1];
        gs->aHep = flow * AlphaHep[j] + fhi * AlphaHep[j + 1];
        gs->aHepp = flow * AlphaHepp[j] + fhi * AlphaHepp[j + 1];
        gs->ad = flow * Alphad[j] + fhi * Alphad[j + 1];
        gs->geH0 = flow * GammaeH0[j] + fhi * GammaeH0[j + 1];
        gs->geHe0 = flow * GammaeHe0[j] + fhi * GammaeHe0[j + 1];
        gs->geHep = flow * GammaeHep[j] + fhi * GammaeHep[j + 1];

        if(gs->necgs <= 1.e-25 || J_UV == 0)
        {
            gs->gJH0ne = gs->gJHe0ne = gs->gJHepne = 0;
        }
        else
        {
            gs->gJH0ne = gJH0 / gs->necgs;
            gs->gJHe0ne = gJHe0 / gs->necgs;
            gs->gJHepne = gJHep / gs->necgs;
        }

        // Calculate H ionization equilibrium
        gs->nH0 = gs->aHp / (gs->aHp + gs->geH0 + gs->gJH0ne);
        gs->nHp = 1.0 - gs->nH0;

        // Calculate He ionization equilibrium
        if((gs->gJHe0ne + gs->geHe0) <= 1.0e-60)
        {
            gs->nHe0 = gs->yhelium;
            gs->nHep = gs->nHepp = 0;
        }
        else
        {
            gs->nHep = gs->yhelium / (1.0 + (gs->aHep + gs->ad) / (gs->geHe0 + gs->gJHe0ne) + (gs->geHep + gs->gJHepne) / gs->aHepp);
            gs->nHe0 = gs->nHep * (gs->aHep + gs->ad) / (gs->geHe0 + gs->gJHe0ne);
            gs->nHepp = gs->nHep * (gs->geHep + gs->gJHepne) / gs->aHepp;
        }

        // Update electron density
        gs->ne = gs->nHp + gs->nHep + 2 * gs->nHepp;
        gs->necgs = gs->ne * gs->nHcgs;

        // Use average of new and old electron density for next iteration
        double nenew = 0.5 * (gs->ne + neold);
        gs->ne = nenew;
        gs->necgs = gs->ne * gs->nHcgs;

        // Check convergence
        if(fabs(gs->ne - neold) < 1.0e-4)
            break;

        neold = gs->ne;
        niter++;
    }
    while(niter < 1000);

    // Calculate bremsstrahlung and other cooling rates
    gs->bH0 = flow * BetaH0[j] + fhi * BetaH0[j + 1];
    gs->bHep = flow * BetaHep[j] + fhi * BetaHep[j + 1];
    gs->bff = flow * Betaff[j] + fhi * Betaff[j + 1];

    *ne_guess = gs->ne;
}

double coolsfr::CoolingRate(double logT, double rho, double *nelec, gas_state *gs, const do_cool_data *DoCool)
{
    double Lambda, Heat;

    if(logT < log10(100.0))
        logT = log10(100.0);

    find_abundances_and_rates(logT, rho, nelec, gs, DoCool);

    double T = pow(10.0, logT);

    // Collisional excitation cooling
    double LambdaExcH0 = gs->bH0 * gs->ne * gs->nH0;
    double LambdaExcHep = gs->bHep * gs->ne * gs->nHep;
    double LambdaExc = LambdaExcH0 + LambdaExcHep;

    // Collisional ionization cooling
    double LambdaIonH0 = 2.18e-11 * gs->geH0 * gs->ne * gs->nH0;
    double LambdaIonHe0 = 3.94e-11 * gs->geHe0 * gs->ne * gs->nHe0;
    double LambdaIonHep = 8.72e-11 * gs->geHep * gs->ne * gs->nHep;
    double LambdaIon = LambdaIonH0 + LambdaIonHe0 + LambdaIonHep;

    // Recombination cooling
    double LambdaRecHp = 1.036e-16 * T * gs->ne * (gs->aHp * gs->nHp);
    double LambdaRecHep = 1.036e-16 * T * gs->ne * (gs->aHep * gs->nHep);
    double LambdaRecHepp = 1.036e-16 * T * gs->ne * (gs->aHepp * gs->nHepp);
    double LambdaRecHepd = 6.526e-11 * gs->ad * gs->ne * gs->nHep;
    double LambdaRec = LambdaRecHp + LambdaRecHep + LambdaRecHepp + LambdaRecHepd;

    // Free-free emission
    double LambdaFF = gs->bff * (gs->nHp + gs->nHep + 4 * gs->nHepp) * gs->ne;

    Lambda = LambdaExc + LambdaIon + LambdaRec + LambdaFF;

    // Compton cooling/heating
    double redshift = 1.0 / All.Time - 1.0;
    double T_CMB = 2.7255 * (1.0 + redshift);
    double LambdaCmptn = 5.65e-36 * gs->ne * (T - T_CMB) * pow(1.0 + redshift, 4) / gs->nHcgs;

    Lambda += LambdaCmptn;

    // Photoheating
    Heat = 0;
    if(J_UV != 0)
        Heat += (gs->nH0 * epsH0 + gs->nHe0 * epsHe0 + gs->nHep * epsHep) / gs->nHcgs;

    // Net cooling rate
    return Heat - Lambda;
}

double coolsfr::CoolingRateFromU(double u, double rho, double *ne_guess, gas_state *gs, const do_cool_data *DoCool)
{
    double temp = convert_u_to_temp(u, rho, ne_guess, gs, DoCool);
    double logT = log10(temp);
    
    return CoolingRate(logT, rho, ne_guess, gs, DoCool);
}

double coolsfr::DoCooling(double u_old, double rho, double dt, double *ne_guess, gas_state *gs, do_cool_data *DoCool)
{
    // Safety checks for invalid input values
    if(!gsl_finite(u_old) || u_old < 0 || rho <= 0 || dt < 0) {
        // Handle the error case by returning a safe value
        if(!gsl_finite(u_old))
            mpi_printf("WARNING: Non-finite u_old=%g detected in DoCooling, returning minimum energy\n", u_old);
        
        // Return minimum allowed energy to avoid termination
        return All.MinEgySpec;
    }

    DoCool->u_old_input = u_old;
    DoCool->rho_input = rho;
    DoCool->dt_input = dt;
    DoCool->ne_guess_input = *ne_guess;

    if(dt == 0)
        return u_old;

    double u = u_old;
    gs->nHcgs = rho * HYDROGEN_MASSFRAC / PROTONMASS;
    double ratefact = gs->nHcgs * gs->nHcgs / rho;

    // Safety check for cooling rate calculation
    double LambdaNet;
    try {
        LambdaNet = CoolingRateFromU(u, rho, ne_guess, gs, DoCool);
        if(!gsl_finite(LambdaNet)) {
            mpi_printf("WARNING: Non-finite LambdaNet in DoCooling, returning original energy\n");
            return u_old;
        }
    }
    catch(...) {
        mpi_printf("WARNING: Exception in CoolingRateFromU, returning original energy\n");
        return u_old;
    }

    // No cooling if net heating
    if(LambdaNet >= 0)
        return u_old;

    // Use semi-implicit integration scheme
    double m_old = 1.0;
    double m = m_old;
    double m_lower = m_old;
    double m_upper = m_old;

    if(m - m_old - m_old * ratefact * LambdaNet * dt > 0)
    {
        // We may want even a smaller timestep
        m_upper *= 1.1;
        m_lower /= 1.1;

        int safety_counter = 0;
        while(m_lower - m_old - m_lower * m_lower / m_old * ratefact * CoolingRateFromU(u, rho * m_lower / m_old, ne_guess, gs, DoCool) * dt > 0)
        {
            m_upper = m_lower;
            m_lower /= 1.1;
            
            // Safety check to avoid infinite loops
            safety_counter++;
            if(m_lower < 1e-10 || safety_counter > 100)
                break;
        }
    }

    // Bisection method to find the new internal energy
    int iter = 0;
    double dm;
    do
    {
        m = 0.5 * (m_lower + m_upper);
        
        // Safety check for cooling rate during bisection
        double tempLambdaNet;
        try {
            tempLambdaNet = CoolingRateFromU(u, rho * m / m_old, ne_guess, gs, DoCool);
            if(!gsl_finite(tempLambdaNet)) {
                mpi_printf("WARNING: Non-finite LambdaNet in bisection, breaking\n");
                break;
            }
        }
        catch(...) {
            mpi_printf("WARNING: Exception in CoolingRateFromU during bisection, breaking\n");
            break;
        }

        if(m - m_old - m * m / m_old * ratefact * tempLambdaNet * dt > 0)
            m_upper = m;
        else
            m_lower = m;

        dm = m_upper - m_lower;
        iter++;

        if(iter >= 150) {
            mpi_printf("slow convergence in DoCooling: m_up=%g, m_low=%g, dm/m=%g\n", m_upper, m_lower, dm / m);
        }
    }
    while(fabs(dm / m) > 1.0e-6 && iter < 100);  // Reduced max iterations for safety

    // Calculate new internal energy
    u = u_old * m / m_old;
    
    // Final safety check for output value
    if(!gsl_finite(u) || u < 0) {
        mpi_printf("WARNING: Invalid energy result in DoCooling u=%g, returning minimum energy\n", u);
        return All.MinEgySpec;
    }
    
    // Limit cooling to reasonable values
    if(u < 0.1 * u_old)
        u = 0.1 * u_old;
    
    // Make sure we don't fall below minimum energy
    if(u < All.MinEgySpec)
        u = All.MinEgySpec;

    return u;
}

void coolsfr::cooling_only(simparticles *Sp)
{
    TIMER_START(CPU_COOLING_SFR);
    
    // Update cosmological factors and ionization parameters
    All.set_cosmo_factors_for_current_time();
    IonizeParams();
    
    gas_state gs = GasState;
    do_cool_data DoCool = DoCoolData;
    
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

void coolsfr::cool_sph_particle(simparticles *Sp, int i, gas_state *gs, do_cool_data *DoCool) {
    // Existing code...
    
    // After cooling, check temperature and apply floor
    double rho = Sp->SphP[i].Density;
    double ne = Sp->SphP[i].Ne;
    double u_old = Sp->get_utherm_from_entropy(i);
    double unew = DoCooling(u_old, rho, dt, &ne, gs, DoCool);
    
    // Convert to temperature
    double temp = convert_u_to_temp(unew, rho, &ne, gs, DoCool);
    
    // Apply temperature floor (adjust this value)
    double min_temp = 50.0; // 10K floor
    if(temp < min_temp) {
        double mean_weight = 4.0 / (1 + 3 * HYDROGEN_MASSFRAC);
        unew = 1.0 / mean_weight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * min_temp;
        unew *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;
    }
     
    // Update particle properties
    Sp->SphP[i].Ne = ne;
    Sp->set_entropy_from_utherm(unew, i);
    Sp->SphP[i].set_thermodynamic_variables();
}

#endif /* COOLING */