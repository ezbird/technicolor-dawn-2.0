/******************************************************************************* 
 *  sfr_eos.cc
 *  
 *  Wrapper for Gadget-4’s isochoric precooling step,
 *  now delegating into the C++ coolsfr class you’ve built.
 *******************************************************************************/

 #include "allvars.h"
 #include "proto.h"
 #include "cosmology.h"
 
 #ifdef COOLING
 #include "cooling.h"
 // Our C++ controller for cooling + star formation
 static coolsfr sfr_module(MPI_COMM_WORLD);
 #endif
 
 /**
  *  Convert internal units to star formation units
  *  (this is the existing Gadget-4 routine; we just add cooling init here)
  */
 void set_units_sfr(void)
 {
     double meanweight;
 
     All.OverDensThresh =
         All.CritOverDensity * All.OmegaBaryon * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.G);
     All.PhysDensThresh =
         All.CritPhysDensity * PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs;
 
     meanweight = 4.0 / (1.0 + 3.0 * HYDROGEN_MASSFRAC);  /* assuming neutral gas */
     All.EgySpecCold =
         (1.0 / meanweight) * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.TempClouds;
     All.EgySpecCold *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;
 
     meanweight = 4.0 / (8.0 - 5.0 * (1.0 - HYDROGEN_MASSFRAC));  /* fully ionized */
     All.EgySpecSN =
         (1.0 / meanweight) * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.TempSupernova;
     All.EgySpecSN *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;
 
 #ifdef COOLING
     // Initialize our cooling tables and UVB photo‐rates
     sfr_module.InitCool();          // loads UV background, sets cosmology factors
     sfr_module.InitCoolMemory();    // allocates RateT[NCOOLTAB+1]
     sfr_module.MakeCoolingTable();  // fills in TREECOOL rates :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
     // Note: we leave init_clouds() in the SF routines (called below)
 #endif
 }
 
 /**
  *  Isochoric pre-cooling step.
  *  We now hand off *all* cooling to coolsfr::cooling_only(), which
  *  internally calls your new DoCooling / find_abundances_and_rates loop.
  */
 void isochoric_precooling(double *uoldArr,
                           double *unewArr,
                           double *duadbArr,
                           double *dtimeArr,
                           double dtimeSys,
                           double a3inv)
 {
 #ifdef COOLING
     // Delegate every gas‐particle to the new C++ cooling routine
     sfr_module.cooling_only(Sp);
 #else
     // Fallback to the original Gadget-4 code
     int i;
     double ne, dtimePart, Zcool = 0.0;
 
     for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
     {
         if(P[i].Type != 0) continue;
 
 #ifdef SFR
         if(SphP[i].SfFlag != 0) continue;
 #endif
         if(!((dtimePart = dtimeArr[i] - dtimeSys) > 0.0))
         {
             unewArr[i] = uoldArr[i];
             continue;
         }
 
 #if defined(METALCOOL) && defined(METALS)
         Zcool = getZ(i);
 #endif
         ne = SphP[i].Ne;
 
         // Original DoCooling signature; now hidden behind the C++ class
         unewArr[i] = DoCooling(uoldArr[i],
                               duadbArr[i],
                               SphP[i].d.Density * a3inv,
                               Zcool,
                               dtimePart,
                               &ne,
                               i);
 
         uoldArr[i] = unewArr[i];
     }
 #endif
 }
 