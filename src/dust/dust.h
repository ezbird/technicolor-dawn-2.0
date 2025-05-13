/*
 =====================================================================================
 DUST.H — On-the-fly Dust Evolution Model for Gadget-4 (Header)
 =====================================================================================
 */

#ifndef DUST_H
#define DUST_H

#include "gadgetconfig.h"

#ifdef DUST

// Add to sphparticledata in dtypes.h
struct dust_data {
    double DustMass;      // Total dust mass
    double SilicateMass;  // Silicate dust mass
    double CarbonMass;    // Carbonaceous dust mass
    double IronMass;      // Iron dust mass
    double GrainSize;     // Mean grain size in μm
};

// Function prototypes
void initialize_dust(simparticles* Sp);

void process_dust_production(simparticles* Sp, int index, int feedback_type, 
                            double metals_added[4], double metallicity);

void process_dust_destruction(simparticles* Sp, int index, double v_kick);

void process_thermal_sputtering(simparticles* Sp, double dt);

void process_dust_growth(simparticles* Sp, double dt);

void process_dust_physics(simparticles* Sp, double dt);

void output_dust_diagnostics(simparticles* Sp);

#endif // DUST

#endif // DUST_H