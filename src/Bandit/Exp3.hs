{-# LANGUAGE DerivingVia #-}

-- |
-- Module      : Bandit.Exp3
-- Copyright   : (c) 2019, UChicago Argonne, LLC.
-- License     : MIT
-- Maintainer  : fre@freux.fr
--
-- The exponential-weight algorithm for Exploration and Exploitation (EXP3). See [1]
--
-- - [1] Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems,
--   Sebastien Bubeck and Nicolo Cesa-Bianchi. http://arxiv.org/abs/1204.5721
module Bandit.Exp3
  ( -- * State
    Exp3 (..),
    exp3,

    -- * Internal
    Weight (..),
    Probability (..),
    CumulativeLoss (..),
  )
where

import Bandit.Class
import Bandit.Types
import Bandit.Util
import Control.Lens hiding (_Unwrapped)
import Data.Generics.Labels ()
import Data.Generics.Wrapped
import Protolude
import qualified Refined as R
import System.Random

-- | The EXP3 state
data Exp3 a = Exp3
  { t :: Int,
    lastAction :: a,
    k :: Int,
    weights :: NonEmpty (Weight a)
  }
  deriving (Show, Generic)

-- | Probability of picking an action
newtype Probability = Probability {getProbability :: Double}
  deriving (Show, Generic)

-- | Cumulative loss counter for an action
newtype CumulativeLoss = CumulativeLoss {getCumulativeLoss :: Double}
  deriving (Show, Generic)

-- | Exp3 weight for one action
data Weight a = Weight
  { probability :: Probability,
    cumulativeLoss :: CumulativeLoss,
    action :: a
  }
  deriving (Show, Generic)

-- | The Exponential-weight algorithm for Exploration and Exploitation (EXP3).
exp3 :: (Eq a) => Bandit (Exp3 a) (Arms a) a (ZeroOne Double)
exp3 =
  Bandit
    { init = \g (Arms as) ->
        let awl = as <&> (Bandit.Types.one,)
            (a, g') = sampleWL awl g
            ws =
              as
                <&> Weight
                  (Probability $ 1.0 / fromIntegral (length (toList as)))
                  (CumulativeLoss 0)
         in ( Exp3
                { t = 1,
                  lastAction = a,
                  k = length as,
                  weights = ws
                },
              a,
              g'
            ),
      step = \g (R.unrefine -> l) -> do
        oldAction <- use #lastAction
        #weights %= fmap (\w -> if action w == oldAction then updateCumLoss l w else w)
        t <- use #t
        k <- use #k
        #weights %= recompute t k
        #t += 1
        pickAction g
    }

pickAction :: (RandomGen g, MonadState (Exp3 a) m) => g -> m (a, g)
pickAction g = do
  bandit <- get
  let (a, g') =
        sampleWL
          ( normalizeDistribution (weights bandit <&> w2tuple) & \case
              Nothing -> panic "Exp3 internal distribution normalization error"
              Just x -> x
          )
          g
  #lastAction .= a
  return (a, g')
  where
    w2tuple :: forall b. Weight b -> (Double, b)
    w2tuple (Weight p _ action) = (getProbability p, action)

updateCumLoss :: Double -> Weight a -> Weight a
updateCumLoss l w@(Weight (Probability p) _ _) =
  w & #cumulativeLoss . _Unwrapped +~ l / p

recompute :: Int -> Int -> NonEmpty (Weight a) -> NonEmpty (Weight a)
recompute t k weights = updatep <$> weights
  where
    updatep w@(Weight _ (CumulativeLoss cL) _) =
      w & #probability . #getProbability
        .~ expw cL
        / denom
    expw cL =
      exp (- sqrt (2.0 * log (fromIntegral k) / fromIntegral (t * k)) * cL)
    denom = getSum $ foldMap denomF weights
    denomF (getCumulativeLoss . cumulativeLoss -> cL) = Sum $ expw cL
