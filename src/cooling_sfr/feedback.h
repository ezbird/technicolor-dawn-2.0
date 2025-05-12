/*
 ==========================================================================================
 FEEDBACK_TREEWALK.H — Stellar Feedback for Gadget-4 (Header)
 ==========================================================================================
 
 ❖ Purpose:
 Header file for the stellar feedback implementation in Gadget-4.
 Includes declarations for both treewalk and brute-force implementations.
 
 ==========================================================================================
 */
 
 #ifndef FEEDBACK_TREEWALK_H
 #define FEEDBACK_TREEWALK_H
 
 #include "../data/simparticles.h"
 
 // Forward declarations
 class FeedbackTreeWalk;
 
 // External variables for energy conversion
 extern double erg_per_mass_to_code;
 
 // Main entry points for feedback
 void apply_stellar_feedback_treewalk(double current_time, simparticles* Sp);
 void apply_stellar_feedback_bruteforce(double current_time, simparticles* Sp);
 
 // Helper function for processing a specific feedback type
 void process_feedback_type(int feedback_type, double current_time, simparticles* Sp, FeedbackTreeWalk* walker);
 
 // Function called from simulation run loop
 // void run_feedback(double current_time, int feedback_type, simparticles* Sp);
 
 void OutputFeedbackDiagnostics();

 #endif // FEEDBACK_TREEWALK_H