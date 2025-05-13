/*
 =====================================================================================
 DUST.CC — On-the-fly Dust Evolution Model for Gadget-4
 =====================================================================================
 
 ❖ Purpose:
 This file implements an on-the-fly dust evolution model that tracks the formation,
 growth, and destruction of dust in cosmological simulations. The model accounts for:
   1. Dust production from different stellar feedback mechanisms (SNII, SNIa, AGB)
   2. Dust growth in the ISM through accretion of metals
   3. Dust destruction by shocks and thermal sputtering
   4. Dust transport and mixing
 
 ❖ Dust Species:
 The model tracks multiple dust species separately:
   • Silicates (predominantly from SNII)
   • Carbonaceous dust (predominantly from AGB stars)
   • Iron-rich dust (predominantly from SNIa)
 
 =====================================================================================
 */
 
#include "gadgetconfig.h"

#ifdef DUST

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <random>
#include <fstream>
#include <iomanip>

#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/mymalloc.h"
#include "../logs/logs.h"
#include "../cooling_sfr/feedback.h"
#include "../sph/sph.h"
#include "dust.h"

// ─── For diagnostic output CSV file ───
std::vector<double> g_dust_silicate;
std::vector<double> g_dust_carbon;
std::vector<double> g_dust_iron;
std::vector<double> g_dust_dtm_ratio;
std::vector<double> g_dust_temp;
std::vector<double> g_dust_dens;
std::vector<double> g_dust_depletion;

// Define constants for dust physics
// ─────────────────────────────────────────────────────

// Dust formation efficiencies by feedback type
// (Condensation efficiencies - what fraction of metals goes into dust)
const double DUST_EFF_SNII_SILICATE = 0.15;  // 15% of SNII metals go into silicate dust
const double DUST_EFF_SNII_CARBON = 0.05;    // 5% of SNII metals go into carbon dust
const double DUST_EFF_SNII_IRON = 0.01;      // 1% of SNII metals go into iron dust

const double DUST_EFF_AGB_SILICATE = 0.10;   // 10% of AGB metals go into silicate dust
const double DUST_EFF_AGB_CARBON = 0.30;     // 30% of AGB metals go into carbon dust
const double DUST_EFF_AGB_IRON = 0.01;       // 1% of AGB metals go into iron dust

const double DUST_EFF_SNIA_SILICATE = 0.02;  // 2% of SNIa metals go into silicate dust
const double DUST_EFF_SNIA_CARBON = 0.01;    // 1% of SNIa metals go into carbon dust
const double DUST_EFF_SNIA_IRON = 0.15;      // 15% of SNIa metals go into iron dust

// Dust growth parameters
const double DUST_GROWTH_TIMESCALE_NORM = 1.0e8;  // years
const double DUST_GROWTH_TEMP_MAX = 300.0;         // K
const double DUST_GROWTH_DENS_MIN = 1.0e-24;       // g/cm³

// Dust destruction parameters
const double DUST_DESTROY_VKICK_THRESHOLD = 50.0;  // km/s
const double DUST_DESTROY_MAX_FRACTION = 0.9;     // Maximum fraction that can be destroyed at once
const double DUST_THERMAL_SPUTTER_TEMP = 1.0e6;    // K

// Dust timing parameters
const double DUST_PROCESSING_INTERVAL = 0.01;     // Process dust every 1% of simulation time

// Diagnostic output frequency
const int DUST_DIAG_FREQUENCY = 10;               // Output every 10 steps or snapshots

/**
 * @brief Initialize dust properties for all gas particles
 * 
 * This function allocates and initializes dust properties for all gas
 * particles in the simulation. It should be called during simulation
 * initialization.
 */
void initialize_dust(simparticles* Sp) {
    if (ThisTask == 0)
        printf("[Dust] Initializing dust model...\n");
    
    for (int i = 0; i < Sp->NumPart; i++) {
        if (Sp->P[i].getType() != 0)  // Skip non-gas particles
            continue;
        
        // Initialize total dust mass
        Sp->SphP[i].DustMass = 0.0;
        
        // Initialize dust species
        Sp->SphP[i].SilicateMass = 0.0;
        Sp->SphP[i].CarbonMass = 0.0;
        Sp->SphP[i].IronMass = 0.0;
        
        // Initialize mean grain size (in μm)
        Sp->SphP[i].GrainSize = 0.1;  // Start with typical 0.1 μm grains
        
        // If we already have metals, initialize with a fraction in dust
        if (Sp->SphP[i].Metals[0] > 0) {
            double metal_mass = Sp->SphP[i].Metals[0] * Sp->P[i].getMass();
            double initial_dust_fraction = 0.1;  // 10% of metals start as dust
            
            double dust_mass = metal_mass * initial_dust_fraction;
            Sp->SphP[i].DustMass = dust_mass;
            
            // Distribute among species
            Sp->SphP[i].SilicateMass = dust_mass * 0.6;  // 60% silicates
            Sp->SphP[i].CarbonMass = dust_mass * 0.3;    // 30% carbon
            Sp->SphP[i].IronMass = dust_mass * 0.1;      // 10% iron
        }
    }
    
    if (ThisTask == 0)
        printf("[Dust] Dust model initialized successfully.\n");
}

/**
 * @brief Process dust production from feedback events
 * 
 * This function should be called after feedback is applied to convert
 * a fraction of the newly injected metals into dust based on the
 * feedback type.
 * 
 * @param Sp Simparticles structure with gas and star particles
 * @param index Gas particle index
 * @param feedback_type Type of feedback (FEEDBACK_SNII, FEEDBACK_SNIa, FEEDBACK_AGB)
 * @param metals_added Array of metal masses added
 * @param metallicity Current metallicity of the gas
 */
void process_dust_production(simparticles* Sp, int index, int feedback_type, 
                             double metals_added[4], double metallicity) {
    // Skip invalid particles
    if (index < 0 || index >= Sp->NumPart || Sp->P[index].getType() != 0)
        return;
    
    double dust_silicate = 0.0;
    double dust_carbon = 0.0;
    double dust_iron = 0.0;
    
    // Calculate dust production based on feedback type
    if (feedback_type == FEEDBACK_SNII) {
        // Type II supernovae - primarily silicates
        dust_silicate = metals_added[0] * DUST_EFF_SNII_SILICATE;
        dust_carbon = metals_added[1] * DUST_EFF_SNII_CARBON;  // Using carbon yield
        dust_iron = metals_added[3] * DUST_EFF_SNII_IRON;     // Using iron yield
    }
    else if (feedback_type == FEEDBACK_AGB) {
        // AGB stars - primarily carbon dust
        dust_silicate = metals_added[0] * DUST_EFF_AGB_SILICATE;
        dust_carbon = metals_added[1] * DUST_EFF_AGB_CARBON;   // Using carbon yield
        dust_iron = metals_added[3] * DUST_EFF_AGB_IRON;      // Using iron yield
    }
    else if (feedback_type == FEEDBACK_SNIa) {
        // Type Ia supernovae - primarily iron dust
        dust_silicate = metals_added[0] * DUST_EFF_SNIA_SILICATE;
        dust_carbon = metals_added[1] * DUST_EFF_SNIA_CARBON;   // Using carbon yield
        dust_iron = metals_added[3] * DUST_EFF_SNIA_IRON;      // Using iron yield
    }
    
    // Scale dust production with metallicity
    // At very low Z, dust formation is less efficient
    double z_factor = pow(fmax(metallicity, 1.0e-6) / 0.02, 0.3);
    dust_silicate *= z_factor;
    dust_carbon *= z_factor;
    dust_iron *= z_factor;
    
    // Update dust masses
    Sp->SphP[index].SilicateMass += dust_silicate;
    Sp->SphP[index].CarbonMass += dust_carbon;
    Sp->SphP[index].IronMass += dust_iron;
    
    // Update total dust mass
    Sp->SphP[index].DustMass = Sp->SphP[index].SilicateMass + 
                               Sp->SphP[index].CarbonMass + 
                               Sp->SphP[index].IronMass;
    
    // Ensure dust mass doesn't exceed metal mass
    double metal_mass = Sp->SphP[index].Metals[0] * Sp->P[index].getMass();
    if (Sp->SphP[index].DustMass > metal_mass) {
        // Scale down each component proportionally
        double scale = metal_mass / Sp->SphP[index].DustMass;
        Sp->SphP[index].SilicateMass *= scale;
        Sp->SphP[index].CarbonMass *= scale;
        Sp->SphP[index].IronMass *= scale;
        Sp->SphP[index].DustMass = metal_mass;
        
        if (ThisTask == 0 && All.HighVerbosity)
            printf("[Dust] Capped dust mass to metal mass for gas ID=%llu\n", 
                  (unsigned long long)Sp->P[index].ID.get());
    }
}

/**
 * @brief Process dust destruction from shocks
 * 
 * This function should be called when a gas particle receives feedback
 * to model the destruction of dust by shock waves.
 * 
 * @param Sp Simparticles structure with gas particles
 * @param index Gas particle index
 * @param v_kick Velocity kick received from feedback
 */
void process_dust_destruction(simparticles* Sp, int index, double v_kick) {
    // Skip invalid particles
    if (index < 0 || index >= Sp->NumPart || Sp->P[index].getType() != 0)
        return;
    
    // Only destroy dust if the velocity kick is above threshold
    if (v_kick < DUST_DESTROY_VKICK_THRESHOLD)
        return;
    
    // Calculate destruction fraction based on velocity kick
    double destroy_fraction = 0.1 * (v_kick / 100.0);  // 10% at 100 km/s, scaled
    
    // Cap at maximum destruction fraction
    destroy_fraction = fmin(destroy_fraction, DUST_DESTROY_MAX_FRACTION);
    
    // Apply destruction to each dust component
    double preserved_fraction = 1.0 - destroy_fraction;
    
    Sp->SphP[index].SilicateMass *= preserved_fraction;
    Sp->SphP[index].CarbonMass *= preserved_fraction;
    Sp->SphP[index].IronMass *= preserved_fraction;
    
    // Update total dust mass
    Sp->SphP[index].DustMass = Sp->SphP[index].SilicateMass + 
                               Sp->SphP[index].CarbonMass + 
                               Sp->SphP[index].IronMass;
    
    // Optionally record diagnostic information
    if (ThisTask == 0 && All.HighVerbosity) {
        printf("[Dust] Destroyed %.1f%% of dust by shock v=%.1f km/s for gas ID=%llu\n", 
              destroy_fraction * 100.0, v_kick, (unsigned long long)Sp->P[index].ID.get());
    }
}

/**
 * @brief Process thermal sputtering of dust
 * 
 * This function models the destruction of dust by thermal sputtering
 * in hot gas. It should be called regularly during the simulation.
 * 
 * @param Sp Simparticles structure
 * @param dt Time step in internal units
 */
void process_thermal_sputtering(simparticles* Sp, double dt) {
    // Convert time step to years for sputtering calculation
    double dt_years = dt * All.UnitTime_in_s / (3600.0 * 24.0 * 365.25);
    
    for (int i = 0; i < Sp->NumPart; i++) {
        if (Sp->P[i].getType() != 0)  // Skip non-gas particles
            continue;
        
        // Get gas temperature
        double temp = Sp->SphP[i].Temperature;
        
        // Thermal sputtering only effective in hot gas
        if (temp < DUST_THERMAL_SPUTTER_TEMP)
            continue;
        
        // Calculate sputtering timescale (in years)
        // Based on Tsai & Mathews (1995)
        double n_H = Sp->SphP[i].Density / (1.0e-24 * 1.67e-24);  // Hydrogen number density in cm^-3
        double a_grain = Sp->SphP[i].GrainSize;  // Grain size in μm
        
        // Timescale in years
        double tau_sputter = 2.0e6 * a_grain / n_H * pow(DUST_THERMAL_SPUTTER_TEMP / temp, 1.5);
        
        // Fraction destroyed in this time step
        double destroy_fraction = dt_years / tau_sputter;
        destroy_fraction = fmin(destroy_fraction, DUST_DESTROY_MAX_FRACTION);
        
        if (destroy_fraction > 0.001) {  // Only process if non-negligible
            double preserved_fraction = 1.0 - destroy_fraction;
            
            // Apply sputtering to each dust component
            Sp->SphP[i].SilicateMass *= preserved_fraction;
            Sp->SphP[i].CarbonMass *= preserved_fraction;
            Sp->SphP[i].IronMass *= preserved_fraction;
            
            // Update total dust mass
            Sp->SphP[i].DustMass = Sp->SphP[i].SilicateMass + 
                                   Sp->SphP[i].CarbonMass + 
                                   Sp->SphP[i].IronMass;
            
            // Update grain size due to sputtering
            // As grains get smaller, sputtering becomes more effective
            Sp->SphP[i].GrainSize *= pow(preserved_fraction, 0.3);
        }
    }
}

/**
 * @brief Process dust growth by accretion of metals
 * 
 * This function models the growth of dust through accretion of metals
 * in the ISM. It should be called regularly during the simulation.
 * 
 * @param Sp Simparticles structure
 * @param dt Time step in internal units
 */
void process_dust_growth(simparticles* Sp, double dt) {
    // Convert time step to years for growth calculation
    double dt_years = dt * All.UnitTime_in_s / (3600.0 * 24.0 * 365.25);
    
    for (int i = 0; i < Sp->NumPart; i++) {
        if (Sp->P[i].getType() != 0)  // Skip non-gas particles
            continue;
        
        // Get gas properties
        double temp = Sp->SphP[i].Temperature;
        double dens = Sp->SphP[i].Density;
        double gas_mass = Sp->P[i].getMass();
        
        // Growth only occurs in cold, dense gas
        if (temp > DUST_GROWTH_TEMP_MAX || dens < DUST_GROWTH_DENS_MIN)
            continue;
        
        // Calculate growth timescale scaled by density and temperature
        double t_grow = DUST_GROWTH_TIMESCALE_NORM * 
                       (DUST_GROWTH_TEMP_MAX / temp) * 
                       (DUST_GROWTH_DENS_MIN / dens);
        
        // Get metal mass available for growth
        double metal_mass = Sp->SphP[i].Metals[0] * gas_mass;
        double dust_mass = Sp->SphP[i].DustMass;
        double available_metals = metal_mass - dust_mass;
        
        // Only grow if there are available metals
        if (available_metals > 0) {
            // Calculate amount of growth in this time step
            // Growth rate is proportional to available metals and inversely
            // proportional to growth timescale
            double growth_fraction = dt_years / t_grow;
            growth_fraction = fmin(growth_fraction, 0.1);  // Limit to 10% per step
            
            double dust_add = available_metals * growth_fraction;
            
            // Calculate how much of each metal to add to each dust species
            // Based on existing ratios of dust species and metal abundances
            double total_dust = dust_mass + 1.0e-10;  // Avoid division by zero
            double silicate_frac = Sp->SphP[i].SilicateMass / total_dust;
            double carbon_frac = Sp->SphP[i].CarbonMass / total_dust;
            double iron_frac = Sp->SphP[i].IronMass / total_dust;
            
            // If no dust exists yet, use default ratios
            if (dust_mass < 1.0e-10) {
                silicate_frac = 0.6;  // 60% silicates
                carbon_frac = 0.3;    // 30% carbon
                iron_frac = 0.1;      // 10% iron
            }
            
            // Add to each species
            Sp->SphP[i].SilicateMass += dust_add * silicate_frac;
            Sp->SphP[i].CarbonMass += dust_add * carbon_frac;
            Sp->SphP[i].IronMass += dust_add * iron_frac;
            
            // Update total dust mass
            Sp->SphP[i].DustMass = Sp->SphP[i].SilicateMass + 
                                  Sp->SphP[i].CarbonMass + 
                                  Sp->SphP[i].IronMass;
            
            // Update grain size due to growth (very slight increase)
            Sp->SphP[i].GrainSize *= (1.0 + growth_fraction * 0.01);
        }
    }
}

/**
 * @brief Main dust processing function
 * 
 * This function handles all dust physics for the current time step.
 * It should be called during each active time step.
 * 
 * @param Sp Simparticles structure
 * @param dt Time step in internal units
 */
void process_dust_physics(simparticles* Sp, double dt) {
    static double last_processing_time = -1.0;
    
    // Only process dust at certain intervals to save computation
    if (last_processing_time >= 0 && 
        (All.Time - last_processing_time) < DUST_PROCESSING_INTERVAL * All.TimeBegin)
        return;
    
    // Process dust physics
    process_thermal_sputtering(Sp, dt);
    process_dust_growth(Sp, dt);
    
    // Update last processing time
    last_processing_time = All.Time;
    
    // Output diagnostics occasionally
    static int step_counter = 0;
    if (++step_counter % DUST_DIAG_FREQUENCY == 0)
        output_dust_diagnostics(Sp);
}

/**
 * @brief Output dust diagnostics to CSV
 * 
 * This function outputs detailed dust diagnostics to a CSV file
 * for post-processing and analysis.
 * 
 * @param Sp Simparticles structure
 */
void output_dust_diagnostics(simparticles* Sp) {
    if (ThisTask != 0)
        return;
    
    // Clear any previous diagnostics
    g_dust_silicate.clear();
    g_dust_carbon.clear();
    g_dust_iron.clear();
    g_dust_dtm_ratio.clear();
    g_dust_temp.clear();
    g_dust_dens.clear();
    g_dust_depletion.clear();
    
    // Collect diagnostics
    for (int i = 0; i < Sp->NumPart; i++) {
        if (Sp->P[i].getType() != 0)  // Skip non-gas particles
            continue;
        
        double gas_mass = Sp->P[i].getMass();
        double metal_mass = Sp->SphP[i].Metals[0] * gas_mass;
        double dust_mass = Sp->SphP[i].DustMass;
        
        // Calculate dust-to-metal ratio
        double dtm_ratio = (metal_mass > 0) ? dust_mass / metal_mass : 0.0;
        
        // Calculate metal depletion (fraction of metals in dust)
        double depletion = dtm_ratio;
        
        // Store diagnostics
        g_dust_silicate.push_back(Sp->SphP[i].SilicateMass);
        g_dust_carbon.push_back(Sp->SphP[i].CarbonMass);
        g_dust_iron.push_back(Sp->SphP[i].IronMass);
        g_dust_dtm_ratio.push_back(dtm_ratio);
        g_dust_temp.push_back(Sp->SphP[i].Temperature);
        g_dust_dens.push_back(Sp->SphP[i].Density);
        g_dust_depletion.push_back(depletion);
    }
    
    // Write to file
    static bool firstCall = true;
    std::ofstream out;
    
    if (firstCall) {
        out.open("dust_diagnostics.csv", std::ios::out);
        firstCall = false;
        
        if (!out.is_open()) {
            fprintf(stderr, "[Error] Could not open dust_diagnostics.csv for writing\n");
            return;
        }
        
        // Write header
        out << "# Dust diagnostics at time=" << All.Time << "\n";
        out << "#silicate,carbon,iron,dtm_ratio,temp,dens,depletion\n";
    } else {
        out.open("dust_diagnostics.csv", std::ios::app);
        
        if (!out.is_open()) {
            fprintf(stderr, "[Error] Could not open dust_diagnostics.csv for appending\n");
            return;
        }
    }
    
    out << std::scientific << std::setprecision(6);
    
    // Write each gas particle's dust diagnostics
    for (size_t i = 0; i < g_dust_silicate.size(); i++) {
        out << g_dust_silicate[i] << ","
            << g_dust_carbon[i] << ","
            << g_dust_iron[i] << ","
            << g_dust_dtm_ratio[i] << ","
            << g_dust_temp[i] << ","
            << g_dust_dens[i] << ","
            << g_dust_depletion[i] << "\n";
    }
    
    out.close();
    
    printf("[Dust] Wrote diagnostics for %zu gas particles at time=%g\n", 
           g_dust_silicate.size(), All.Time);
}

#endif // DUST