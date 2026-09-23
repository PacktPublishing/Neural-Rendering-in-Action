#language slang 2026

module nrc;

// RTXNS is the concrete Chapter 7 backend. Centralizing its imports here makes
// that dependency explicit; the __include declarations below assemble the
// NRC module implementation.
import MLP;
import Activation;
import CooperativeVectorFunctions;
import CooperativeVectorAutoDiff;

__include "nrc/records.slang";
__include "nrc/encoders.slang";
__include "nrc/features.slang";
__include "nrc/estimator.slang";
__include "nrc/network.slang";
