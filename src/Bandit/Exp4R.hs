{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- |
-- Module      : Bandit.Exp4R
-- Copyright   : (c) 2019, UChicago Argonne, LLC.
-- License     : MIT
-- Maintainer  : fre@freux.fr
--
-- The contextual exponential-weight algorithm for Exploration and Exploitation
-- with Experts and Risk Constraints (EXP4R). See [1]
--
-- - [1] Sun, W., Dey, D. & Kapoor, A.. (2017). Safety-Aware Algorithms for
-- Adversarial Contextual Bandit. Proceedings of the 34th International
-- Conference on Machine Learning, in PMLR 70:3280-3288
module Bandit.Exp4R
  ( -- * Interface
    exp4r,
    Feedback (..),

    -- * State
    Exp4R (..),
    LastAction (..),

    -- * Configuration
    Exp4RCfg (..),

    -- * Experts
    ObliviousRep (..),

    -- * internal
    mkMu,
    mkDelta,
    lambdaInitial,
  )
where

import Bandit.Class
import Bandit.Types
import Bandit.Util
import Control.Lens
import Data.Generics.Labels ()
import Data.List.NonEmpty as NE
import Protolude
import qualified Refined as R
import qualified Refined.Unsafe as R

-- | The EXP4R state
data Exp4R s a er = Exp4R
  { t :: Int,
    horizon :: R.Refined R.Positive Int,
    lastAction :: Maybe (LastAction a),
    k :: Int,
    n :: Int,
    lambda :: R.Refined R.NonNegative Double,
    constraint :: ZeroOne Double,
    experts ::
      NonEmpty
        ( ZeroOne Double,
          er
        )
  }
  deriving (Generic)

-- | Encapsulator for 'last action taken'
data LastAction a = LastAction
  { action :: a,
    globalProbabilityOfSample :: ZeroOne Double,
    perExpertProbabilityOfSample :: NonEmpty (ZeroOne Double)
  }
  deriving (Generic)

-- | Constructor for feedback from the environment.
data Feedback = Feedback
  { cost :: ZeroOne Double,
    risk :: ZeroOne Double
  }
  deriving (Generic)

-- | Hyperparameters.
data Exp4RCfg s a er = Exp4RCfg
  { expertsCfg :: NonEmpty er,
    constraintCfg :: ZeroOne Double,
    horizonCfg :: R.Refined R.Positive Int,
    as :: NonEmpty a
  }
  deriving (Generic)

exp4r ::
  (Eq a, ExpertRepresentation er s a) =>
  ContextualBandit (Exp4R s a er) (Exp4RCfg s a er) s a (Maybe Feedback) er
exp4r =
  ContextualBandit
    { initCtx = \Exp4RCfg {..} ->
        Exp4R
          { t = 1,
            lastAction = Nothing,
            k = NE.length as,
            n = NE.length expertsCfg,
            lambda = lambdaInitial,
            constraint = constraintCfg,
            horizon = horizonCfg,
            experts = (R.unsafeRefine (1 / fromIntegral (NE.length expertsCfg)),) <$> expertsCfg
          },
      stepCtx = \g feedback s -> do
        weightedAdvice <- use #experts <&> fmap (fmap (($ s) . toExpert))
        lastAction <- use #lastAction
        fromMaybe
          pass
          (update weightedAdvice <$> lastAction <*> feedback)
        let armDistribution =
              fromMaybe
                (panic "internal Exp4R algorithm failure: distribution normalization failed.")
                (combineAdvice weightedAdvice)
            (a, g') = sampleWL armDistribution g
            p_a =
              maybe
                (panic "internal Exp4R algorithm failure: arm pull issue.")
                fst
                (find (\x -> snd x == a) armDistribution)
            probabilityOf_a :: NonEmpty (ZeroOne Double)
            probabilityOf_a =
              snd <$> weightedAdvice <&> \e ->
                maybe
                  (panic "internal Exp4R algorithm failure: weight computation")
                  fst
                  (find (\x -> snd x == a) e)
        #lastAction ?= LastAction a p_a probabilityOf_a
        return (a, g')
    }

update ::
  (MonadState (Exp4R s a er) m) =>
  NonEmpty (ZeroOne Double, NonEmpty (ZeroOne Double, a)) ->
  LastAction a ->
  Feedback ->
  m ()
update
  weightedAdvice
  (LastAction _ (R.unrefine -> p_a) (fmap R.unrefine -> pPolicy_a))
  (Feedback (R.unrefine -> c) (R.unrefine -> r)) = do
    lam <- R.unrefine <$> use #lambda
    delta <- get <&> mkDelta
    mu <- get <&> mkMu
    beta <- use #constraint <&> R.unrefine
    let numeratorTerm (R.unrefine -> w, _) p =
          w * exp (- mu * (p * (lam * r + c) / p_a))
        wUpdate = NE.zipWith numeratorTerm weightedAdvice pPolicy_a
        wDenom = getSum . sconcat $ Sum <$> wUpdate
    #experts
      %= NE.zipWith (\w' (_, e) -> (unsafeNormalizePanic w' wDenom, e)) wUpdate
    let fDot (R.unrefine -> w, _) p = w * r * p / p_a
    let dotted =
          getSum $ sconcat (Sum <$> NE.zipWith fDot weightedAdvice pPolicy_a)
    #lambda
      .= R.unsafeRefine (max 0 (lam + mu * (dotted - beta - delta * mu * lam)))

-- | combineAdvice turns weighted expert advice into a probability distribution to
-- sample from.
combineAdvice ::
  (Ord a) =>
  NonEmpty (ZeroOne Double, NonEmpty (ZeroOne Double, a)) ->
  Maybe (NonEmpty (ZeroOne Double, a))
combineAdvice weightedAdvice =
  normalizeDistribution $
    groupAllWith1 snd dirtyArmDistribution
      <&> \gs -> (getSum $ sconcat (gs <&> Sum . fst), snd $ NE.head gs)
  where
    dirtyArmDistribution =
      sconcat $
        weightedAdvice
          <&> \(wi, advices) ->
            advices
              <&> \(p, ai) -> (R.unrefine p * R.unrefine wi, ai)

-- | \( \mu = \sqrt{\frac{\ln N }{ (T(K+4))}} \)
mkMu :: Exp4R s a er -> Double
mkMu Exp4R {..} =
  sqrt $ log (fromIntegral n) / fromIntegral (R.unrefine horizon * (k + 4))

-- | \( \delta = 3K \)
mkDelta :: Exp4R s a er -> Double
mkDelta Exp4R {..} = fromIntegral $ 3 * k

-- | \( \lambda_1 = 0 \)
lambdaInitial :: R.Refined R.NonNegative Double
lambdaInitial = R.unsafeRefine 0
