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
 #include "../cooling_sfr/starformation.h"
 #include "../data/allvars.h"
 #include "../data/dtypes.h"
 #include "../data/mymalloc.h"
 #include "../logs/logs.h"
 #include "../logs/timer.h"
 #include "../system/system.h"
 
 extern coolsfr CoolSfr;  // This declares that CoolSfr is defined elsewhere


 /*! \brief Finalize star formation module
  *
  *  This function closes the star formation log file
  *  when the simulation ends.
  */
 void end_starformation(void)
 {
   TIMER_START(CPU_MISC);
 
   printf("STARFORMATION: Ending star formation module...\n");
 
   TIMER_STOP(CPU_MISC);
 }
 
 #endif /* STARFORMATION */
 #endif /* COOLING */