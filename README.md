# ML4MOC
| **Feature**       | **Definition**                                                                                               |
|-------------------|-------------------------------------------------------------------------------------------------------------|
| **~ Matrix ~**    |                                                                                                             |
| Rows              | $\ln(m)$                                                                                                    |
| Columns           | $\ln(n)$                                                                                                    |
| NonZeros          | Ratio of non-zeros, over $m \times n$                                                                       |
| Symmetries        | 1 if any symmetry, 0 otherwise                                                                              |
| **~ Variables ~** |                                                                                                             |
| Binaries          | Ratio of binary variables, over $n$                                                                         |
| Integers          | Ratio of integer variables, over $n$                                                                        |
| **~ Constraints ~** |                                                                                                           |
| LessThan          |                                                                                                             |
| GreaterThan       |                                                                                                             |
| Equality          |                                                                                                             |
| SetPartitioning   |                                                                                                             |
| SetPacking        |                                                                                                             |
| SetCovering       |                                                                                                             |
| Cardinality       |                                                                                                             |
| KnapsackEquality  | Ratio of constraints per constraint type, over $m$                                                          |
| Knapsack          |                                                                                                             |
| KnapsackInteger   |                                                                                                             |
| BinaryPacking     |                                                                                                             |
| VariableLowerBound|                                                                                                             |
| VariableUpperBound|                                                                                                             |
| MixedBinary       |                                                                                                             |
| MixedInteger      |                                                                                                             |
| Continuous        |                                                                                                             |
| **~ Scaling ~**   |                                                                                                             |
| Coefficient_00m   | $\ln(\max A'/ \min A')$                                                                                     |
| RightHandSide_00m | $\ln(\max B'/ \min B')$                                                                                     |
| Objective_00m     | $\ln(\max C'/ \min C')$                                                                                     |
| **~ Presolving ~** |                                                                                                            |
| PresolRows        | $\ln(m)$                                                                                                    |
| PresolColumns     | $\ln(n)$                                                                                                    |
| PresolIntegers    | Ratio of presolved integer variables, over $n$                                                              |
| **~ Global Cutting ~** |                                                                                                        |
| DualInitialGap    | $\frac{|c_d - c_l|}{\max(|c_d|, |c_l|, |c_d - c_l|)}$                                                       |
| PrimalDualGap     | $\frac{|c_p - c_l|}{\max(|c_p|, |c_l|, |c_p - c_l|)}$                                                       |
| PrimalInitialGap  | $\frac{|c_p - c_l|}{\max(|c_p|, |c_l|, |c_p - c_l|)}$                                                       |
| GapClosed         | $1 - \text{PrimalDualGap}$                                                                                  |
