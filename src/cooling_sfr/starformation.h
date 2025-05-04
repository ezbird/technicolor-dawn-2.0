/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file starformation.h
 *
 *  \brief declares interface for star formation module
 */

 #ifndef STARFORMATION_H
 #define STARFORMATION_H
 
 #include "gadgetconfig.h"
 
 #ifdef COOLING
 #ifdef STARFORMATION
 
 /* Forward declaration */
 class simparticles;
 
 // Forward declaration for the simulation class
class sim;

 /* Starformation configuration parameters */
 struct starformation_params
 {
   double CritPhysDensity;
   double CritOverDensity;
   double FactorEVP;
   double FactorSN;
   double TempSupernova;
   double TempClouds;
   double MaxSfrTimescale;
   double WindEfficiency;
   double WindEnergyFraction;
   double WindFreeTravelLength;
   double WindFreeTravelDensFac;
   double TargetGasMass;
   double GInternal;        /* Internal G for some SF models */
   
 };
 
 /* Function prototypes */
 void init_starformation(void);
 void end_starformation(void);
 
 #endif /* STARFORMATION */
 #endif /* COOLING */
 #endif /* STARFORMATION_H */