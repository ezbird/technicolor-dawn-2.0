// src/cooling_sph/sfr_modern.cpp
// Implements the star formation routines as methods of coolsfr

#include "gadgetconfig.h"

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../cooling_sfr/cooling.h"
#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../logs/logs.h"
#include "../system/system.h"
#include "../time_integration/timestep.h"

#ifdef COOLING

// Compute the density threshold and related parameters
void coolsfr::init_clouds()
{
    if(All.PhysDensThresh != 0) return;

    double A0     = All.FactorEVP;
    double egyhot = All.EgySpecSN / A0;

    // compute u4 (internal energy at 1e4 K)
    double meanweight = 4.0 / (8.0 - 5.0 * (1.0 - HYDROGEN_MASSFRAC));
    double u4 = (1.0/meanweight) * (1.0/GAMMA_MINUS1)
              * (BOLTZMANN/PROTONMASS) * 1e4;
    u4 *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

    double dens;
    if(All.ComovingIntegrationOn) {
        dens = 1.0e6 * 3.0 * All.Hubble * All.Hubble / (8.0 * M_PI * All.G);
    } else {
        dens = 9.205e-24;
        dens /= (All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam);
    }

    if(All.ComovingIntegrationOn) {
        All.Time = 1.0;
        IonizeParams();
    }

    double ne = 1.0;
    SetZeroIonization();

    double tcool    = GetCoolingTime(egyhot, dens, &ne, &GasState, &DoCoolData);
    double coolrate = egyhot / tcool / dens;

    double x = (egyhot - u4) / (egyhot - All.EgySpecCold);

    All.PhysDensThresh = x / std::pow(1 - x, 2)
        * (All.FactorSN*All.EgySpecSN - (1 - All.FactorSN)*All.EgySpecCold)
        / (All.MaxSfrTimescale * coolrate);
}

// Determine if particle i is eligible for star formation
bool coolsfr::sf_evaluate_particle(simparticles *Sp, int i)
{
    double dens_cgs = Sp->SphP[i].Density * All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;
    double temp = GasState.mhboltz * (Sp->SphP[i].Entropy * std::pow(Sp->SphP[i].Density, GAMMA_MINUS1))
                / GAMMA_MINUS1;

    if(temp < All.TempSfrThresh && Sp->P[i].Mass > 0) {
        if(dens_cgs >= All.PhysDensThresh) {
            Sp->SphP[i].SfFlag = 1;
            return true;
        }
    }
    Sp->SphP[i].SfFlag = 0;
    return false;
}

// Compute star formation rate for particle i (Msun/yr)
double coolsfr::get_starformation_rate(simparticles *Sp, int i, double *cloudMassFraction)
{
    if(Sp->SphP[i].SfFlag == 0) {
        *cloudMassFraction = 0.0;
        return 0.0;
    }

    double rho_phys = Sp->SphP[i].d.Density * All.cf_a3inv;
    double t_ff     = std::sqrt((3*M_PI)/(32.0 * rho_phys * All.G));
    double tsfr     = t_ff / All.MaxSfrTimescale;

    double cloudmass = Sp->P[i].Mass;
    *cloudMassFraction = cloudmass;

    double sfr = cloudmass / tsfr;
    // convert internal units to Msun/yr
    sfr *= (All.UnitMass_in_g/SOLAR_MASS) / (All.UnitTime_in_s/SEC_PER_YEAR);

    return sfr;
}

// Update particle’s entropy after cooling+SF\ nvoid coolsfr::update_thermodynamic_state(simparticles *Sp, int i, double dt, double cloudMassFraction)
{
    double u_current = Sp->SphP[i].Entropy * std::pow(Sp->SphP[i].Density, GAMMA_MINUS1) / GAMMA_MINUS1;
    Sp->SphP[i].Entropy = u_current * GAMMA_MINUS1 / std::pow(Sp->SphP[i].Density, GAMMA_MINUS1);
    Sp->SphP[i].e.DtEntropy = 0.0;
}

// Loop over all active gas particles: cooling then SF\ nvoid coolsfr::cooling_and_starformation(simparticles *Sp)
{
    for(int p = FirstActiveParticle; p >= 0; p = NextActiveParticle[p]) {
        if(Sp->P[p].Type != 0) continue;

        // cooling
        cool_sph_particle(Sp, p, &GasState, &DoCoolData);

        // star formation
        if(sf_evaluate_particle(Sp, p)) {
            double cloudFrac;
            double sfr = get_starformation_rate(Sp, p, &cloudFrac);
            update_thermodynamic_state(Sp, p, DoCoolData.dt_input, cloudFrac);
            Sp->SphP[p].Sfr = sfr;
        } else {
            Sp->SphP[p].Sfr = 0.0;
        }
    }
}

#endif // COOLING
