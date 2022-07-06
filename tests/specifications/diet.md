# Diet model

## Dimensions

+ $\S^d_{recipes} : R$
+ $\S^d_{nutrients} : N$

## Parameters

+ $\S^p_{minimalNutrients} : m \in \mathbb{R}_+^N$
+ $\S^p_{nutrientsPerRecipe} : p \in \mathbb{R}_+^{N \times R}$
+ $\S^p_{costPerRecipe} : c \in \mathbb{R}_+^R$

## Variables

+ $\S^v_{quantityOfRecipe} : \alpha \in \mathbb{N}^R$

## Objective

Minimize total cost: $\S^o : \min \sum_{r \in R} c_r \alpha_r$

## Constraint

Have at least the minimum quantity of each nutrient:

$$
  \S^c_{enoughNutrients} :
  \forall n \in N,
    \sum_{r \in R} \alpha_r p_{n,r} \geq m_n
$$
