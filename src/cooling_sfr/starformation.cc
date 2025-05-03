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

 /*! \brief Initialize star formation module
  *
  *  This function initializes the star formation module.
  *  It sets up the cooling units, initializes the star formation
  *  log file, and computes the tables for the effective model.
  */
 void init_starformation(sim *Sim)
 {
   TIMER_START(CPU_MISC);
 
   printf("STARFORMATION: Initializing star formation module...\n");
   
   /* Initialize the star formation log file */
   if(ThisTask == 0)
     {
       char buf[MAXLEN_PATH];
       sprintf(buf, "%s/sfr.txt", All.OutputDir);
       FILE *fd;
       if(!(fd = fopen(buf, "w")))
         Terminate("Cannot open file '%s' for writing star formation log.\n", buf);
       
       fprintf(fd, "# Time SFR\n");
       fprintf(fd, "# a    Msun/yr\n");
       fclose(fd);
     }
   
   /* Initialize the multi-phase model for star formation */
   CoolSfr.init_clouds();
 
   TIMER_STOP(CPU_MISC);
 }
 
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