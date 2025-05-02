/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file starformation.cc
 *
 *  \brief Main initialization for star formation module
 */

 #include "gadgetconfig.h"

 #ifdef COOLING
 #ifdef STARFORMATION
 
 #include <mpi.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 
 #include "../cooling_sfr/cooling.h"
 #include "../data/allvars.h"
 #include "../data/dtypes.h"
 #include "../data/simparticles.h"
 #include "../logs/logs.h"
 #include "../system/system.h"
 
 /*! \brief Initialize star formation module
  *
  *  This function initializes the star formation module.
  *  It sets up the cooling units, initializes the star formation
  *  log file, and computes the tables for the effective model.
  */
 void sim::init_starformation(void)
 {
   TIMER_START(CPU_INIT_STARFORMATION);
 
   mpi_printf("STARFORMATION: Initializing star formation module...\n");
 
   /* Set up units for cooling and star formation */
   Mem.CoolSfr->set_units_sfr();
   
   /* Initialize the star formation log file */
   Mem.CoolSfr->init_star_formation_log();
   
   /* Initialize the multi-phase model for star formation */
   Mem.CoolSfr->init_clouds();
 
   TIMER_STOP(CPU_INIT_STARFORMATION);
 }
 
 /*! \brief Finalize star formation module
  *
  *  This function closes the star formation log file
  *  when the simulation ends.
  */
 void sim::end_starformation(void)
 {
   TIMER_START(CPU_MISC);
 
   /* Close the star formation log file */
   Mem.CoolSfr->close_star_formation_log();
 
   TIMER_STOP(CPU_MISC);
 }
 
 #endif /* STARFORMATION */
 #endif /* COOLING */