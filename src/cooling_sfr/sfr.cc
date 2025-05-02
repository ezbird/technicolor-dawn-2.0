/*******************************************************************************
 *  sfr_eos.cc
 *
 *  Combined star formation and cooling integration for Gadget-4
 ******************************************************************************/

#include "allvars.h"
#include "proto.h"
#include "cosmology.h"

#ifdef COOLING
#include "cooling.h"
// C++ controller for cooling + star formation
static coolsfr sfr_module(MPI_COMM_WORLD);
#endif

/**
 *  Convert internal units to star formation units
 *  (existing Gadget-4 conversions)
 */
void set_units_sfr(void)
{
    // e.g., convert internal mass/time to M_sun/yr, etc.
    // All.UnitMass_in_g, All.UnitTime_in_s -> SFR units
}

/**
 *  Initialize the cooling and star formation module
 *  Call once before entering the main timestep loop
 */
void init_sfr_module(void)
{
    set_units_sfr();

#ifdef COOLING
    // Initialize cooling (UVB, cosmology factors)
    sfr_module.InitCool();
    // Allocate and fill the TREECOOL table
    sfr_module.InitCoolMemory();
    sfr_module.MakeCoolingTable();
    // Compute star formation density threshold, etc.
    sfr_module.init_clouds();
#endif
}

/**
 *  Isochoric pre-cooling and star formation step
 *  uoldArr:  old internal energy per mass
 *  unewArr:  new internal energy per mass
 *  duadbArr: adiabatic change term
 *  dtimeArr: timestep array
 *  cf_a3inv:  unit conversion factor a^-3
 */
void isochoric_precooling(double *uoldArr,
                          double *unewArr,
                          double *duadbArr,
                          double *dtimeArr,
                          double cf_a3inv)
{
#ifdef COOLING
    // Delegate the entire cooling+SF loop to the C++ class
    sfr_module.cooling_and_starformation(Sp);
#else
    // Fallback to original Gadget-4 logic (can be removed)
    int i;
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i]) {
        if(P[i].Type != 0)
            continue;

        double uold      = uoldArr[i];
        double rho       = SphP[i].Density * cf_a3inv;
        double dt        = dtimeArr[i];
        double duadb     = duadbArr[i];
        double ne_guess  = 0.0;

        // Original cooling call:
        // double u = DoCooling(uold, rho, dt, &ne_guess, &GasState, &DoCoolData);
        // unewArr[i] = u;
        // duadbArr[i] = duadb;

        // Original star formation logic:
        // if (SphP[i].SfFlag) {
        //     double xcloud;
        //     double sfr = get_starformation_rate(i, &xcloud);
        //     /* spawn stars, update entropy, etc. */
        // }
    }
#endif
}
