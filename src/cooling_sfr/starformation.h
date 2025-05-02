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

void init_starformation(void);
void end_starformation(void);

/* starformation configuration parameters */
struct starformation_params
{
  double CritPhysDensity;
  double CritOverDensity;
  double FactorEVP;
  double FactorSN;
  double TempSupernova;
  double TempClouds;
  double MaxSfrTimescale;
  double TempSfrThresh;
  double WindEfficiency;
  double WindEnergyFraction;
  double WindFreeTravelLength;
  double WindFreeTravelDensFac;
  double TargetGasMass;
  double GInternal;        /* Internal G for some SF models */
  
  /* Additional parameters for H2-regulated star formation */
#ifdef H2REGSF
  double OTUVThresh;       /* Threshold for OTUV field */
  double SIGMA_NORM;       /* Surface density normalization */
#endif
};

/* add variables to class sim */
class sim : public simparticles
{
 private:
#ifdef STARFORMATION
  void init_starformation(void);
  void end_starformation(void);
  
  /* include this in the timebin loop for star formation */
  void do_star_formation_and_cooling(void)
  {
    TIMER_START(CPU_COOLING_SFR);
    Mem.CoolSfr->cooling_and_starformation(this);
    TIMER_STOP(CPU_COOLING_SFR);
    
    /* Rearrange particles to maintain proper order */
    Mem.CoolSfr->rearrange_particle_sequence(this);
  }
#endif
};

#endif /* STARFORMATION */
#endif /* COOLING */
#endif /* STARFORMATION_H */