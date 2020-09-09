{-# OPTIONS_GHC -fno-warn-partial-fields #-}

-- |
-- Module      : Bandit.UCB
-- Copyright   : (c) 2019, UChicago Argonne, LLC.
-- License     : MIT
-- Maintainer  : fre@freux.fr
--
-- This module implements the UCB family of algorithms.
module Bandit.UCB
  ( UCB (..),
    ucb,
    UCBHyper (..),
    hyperAlphaUCB,
    hyperUCB1,
  )
where

import Bandit.Class
import Bandit.EpsGreedy
import Bandit.Types
import Control.Lens
import Data.Generics.Labels ()
import Data.List.Extras.Argmax
import Protolude
import Refined

-- | Hyperparameter for \((\alpha,\phi)\)-UCB
data UCBHyper a r = UCBHyper
  { invLFPhiUCB :: r,
    alphaUCB :: Double,
    armsUCB :: Arms a
  }
  deriving (Show, Generic)

-- | Hyperparameter for \(\alpha\)-UCB
hyperAlphaUCB :: Double -> Arms a -> UCBHyper a AlphaUCBInvLFPhi
hyperAlphaUCB = UCBHyper AlphaUCBInvLFPhi

-- | Hyperparameter for parameter-free UCB1
hyperUCB1 :: Arms a -> UCBHyper a AlphaUCBInvLFPhi
hyperUCB1 = hyperAlphaUCB 4

-- | State for the UCB algorithm.
data UCB a p = UCB
  { t :: Int,
    invLFPhi :: p,
    alpha :: Double,
    lastAction :: a,
    params :: Params a
  }
  deriving (Show, Generic)

toW :: (Double, a) -> Weight a
toW (loss, action) = Weight loss 1 action

-- | The variable rate \(\epsilon\)-Greedy MAB algorithm.
-- Offers no interesting guarantees, works well in practice.
ucb :: (Eq a, InvLFPhi p) => Bandit (UCB a p) (UCBHyper a p) a (ZeroOne Double)
ucb =
  Bandit
    { init = \g (UCBHyper invLFPhiUCB alphaUCB (Arms (a :| as))) ->
        ( UCB
            { t = 1,
              alpha = alphaUCB,
              invLFPhi = invLFPhiUCB,
              lastAction = a,
              params =
                InitialScreening $
                  Screening
                    { screened = [],
                      screenQueue = as
                    }
            },
          a,
          g
        ),
      step = \g (unrefine . rewardCostBijection -> l) -> do
        oldAction <- use #lastAction
        invLFPhiFunc <- use #invLFPhi <&> toInvLFPhi
        alphaValue <- use #alpha
        #t += 1
        t <- use #t
        a <-
          use #params >>= \case
            InitialScreening sg ->
              case screenQueue sg of
                (a : as) -> do
                  #params . #_InitialScreening
                    .= Screening
                      { screened = (l, oldAction) : screened sg,
                        screenQueue = as
                      }
                  return a
                [] -> do
                  let ee =
                        ExploreExploit
                          { weights = toW <$> ((l, oldAction) :| screened sg)
                          }
                  #params .= Started ee
                  pickreturn t invLFPhiFunc alphaValue ee
            Started ee -> do
              let ee' = ee & #weights %~ updateWeight oldAction l
              #params .= Started ee'
              pickreturn t invLFPhiFunc alphaValue ee'
        #lastAction .= a
        return (a, g)
    }

--- | Action selection and  return
pickreturn ::
  (MonadState (UCB a r) m) =>
  Int ->
  (Double -> Double) ->
  Double ->
  ExploreExploit a ->
  m a
pickreturn t phiInv alpha (ExploreExploit weights) =
  return . action . argmax f $ toList weights
  where
    f Weight {..} =
      averageLoss + phiInv (alpha * log (fromIntegral t) / fromIntegral hits)
