# Ali Rad - Target Localization Method Comparison Statistics for aim 1

import math

# in mm
navError = 0.75 # "neuronavigation error, brainsights own localization/registration noise"
# double check navError against the brainsight manual

# thresholds
bias = 1.5 # smallest average offset (for t-test)
eqMargin = 2.0 # size of difference to be equiavlent (for one sided test)
LOAPrec = 1.0 # limit of agreement precision (for Bland-Altman)

# sweep the cap drawing error over 1, 2, 3 mm
capErrors = [1.0, 2.0, 3.0]

for capError in capErrors:
    spread = (capError**2 + navError**2) ** 0.5

    nTtest = (1.96 + 0.84)**2 * spread**2 / bias**2
    nTost = (1.645 + 1.28)**2 * spread**2 / eqMargin**2
    nBland = (1.96 * 1.71 * spread / LOAPrec)**2

    print()
    print("cap error:", capError, "mm")
    print("t-test n:", math.ceil(nTtest)) #math.ceil to round these up
    print("TOST n:", math.ceil(nTost))
    print("Bland-Altman n:", math.ceil(nBland))

print()

## there's also packages that can do this, but I did it this to way learn it
