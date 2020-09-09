-- |
-- Module      : Bandit.EpsGreedy
-- Copyright   : (c) 2019, UChicago Argonne, LLC.
-- License     : MIT
-- Maintainer  : fre@freux.fr
--
-- This module implements the fixed rate \(\epsilon\)-Greedy MAB algorithm.
--
-- The \(\epsilon\)-Greedy algorithm selects a random action with
-- probability \(\epsilon\), or select the action with best average
-- with probability \(1-\epsilon\).
module Bandit.EpsGreedy
  ( EpsGreedy (..),
    epsGreedy,
    Weight (..),
    Screening (..),
    EpsGreedyHyper (..),
    Params (..),
    ExploreExploit (..),
    pickRandom,
    updateAvgLoss,
    updateWeight,
  )
where

import Bandit.Class
import Bandit.Types
import Bandit.Util
import Control.Lens
import Control.Monad.Random as MR (fromList, runRand)
import Data.Generics.Labels ()
import Protolude
import System.Random

-- | The state holder for the \(\epsilon\)-Greedy algorithm.
data EpsGreedy a r = EpsGreedy
  { t :: Int,
    rate :: r,
    lastAction :: a,
    params :: Params a
  }
  deriving (Show, Generic)

-- | Params a is used to distinguish between initial screening for values
-- and an ongoing exploration/exploitation process.
data Params a
  = InitialScreening (Screening a)
  | Started (ExploreExploit a)
  deriving (Show, Generic)

-- | Still screening for initial estimates
data Screening a = Screening
  { screened :: [(Double, a)],
    screenQueue :: [a]
  }
  deriving (Show, Generic)

-- | The sampling procedure has started.
newtype ExploreExploit a = ExploreExploit {weights :: NonEmpty (Weight a)}
  deriving (Show, Generic)

-- | The information maintaining structure for one action.
data Weight a = Weight
  { averageLoss :: Double,
    hits :: Int,
    action :: a
  }
  deriving (Show)
  deriving (Generic)

toW :: (Double, a) -> Weight a
toW (loss, action) = Weight loss 1 action

-- | The epsilon-greedy hyperparameter.
data EpsGreedyHyper a r = EpsGreedyHyper
  { rateRep :: r,
    arms :: Arms a
  }
  deriving (Show)

-- | The variable rate \(\epsilon\)-Greedy MAB algorithm.
-- Offers no interesting guarantees, works well in practice.
epsGreedy ::
  (Eq a, Rate r) =>
  Bandit
    (EpsGreedy a r)
    (EpsGreedyHyper a r)
    a
    Double
epsGreedy =
  Bandit
    { init = \g (EpsGreedyHyper r (Arms (a :| as))) ->
        ( EpsGreedy
            { t = 1,
              rate = r,
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
      step = \g l -> do
        oldAction <- use #lastAction
        schedule <- use #rate <&> toRate
        e <- use #t <&> schedule
        (a, newGen) <-
          use #params >>= \case
            InitialScreening sg ->
              case screenQueue sg of
                (a : as) -> do
                  #params
                    .= InitialScreening
                      ( Screening
                          { screened = (l, oldAction) : screened sg,
                            screenQueue = as
                          }
                      )
                  return (a, g)
                [] -> do
                  let ee =
                        ExploreExploit
                          { weights = toW <$> ((l, oldAction) :| screened sg)
                          }
                  #params .= Started ee
                  pickreturn e g ee
            Started ee -> do
              let ee' = ee & #weights %~ updateWeight oldAction l
              #params . #_Started .= ee'
              pickreturn e g ee
        #lastAction .= a
        #t += 1
        return (a, newGen)
    }

-- | Action selection and  return
pickreturn ::
  (RandomGen g, MonadState (EpsGreedy b r) m) =>
  Double ->
  g ->
  ExploreExploit b ->
  m (b, g)
pickreturn eps g eeg = do
  let (a, g') =
        runRand (MR.fromList [(True, toRational eps), (False, toRational $ 1 - eps)]) g & \case
          (True, g'') -> pickRandom eeg g''
          (False, g'') -> (action $ minimumBy (\(averageLoss -> a1) (averageLoss -> a2) -> compare a1 a2) (weights eeg), g'')
  return (a, g')

-- | Random action selection primitive
pickRandom :: (RandomGen g) => ExploreExploit a -> g -> (a, g)
pickRandom ExploreExploit {..} =
  sampleWL $
    fromMaybe
      (panic "distribution normalization failure")
      (normalizeDistribution $ weights <&> w2tuple)
  where
    w2tuple :: Weight b -> (Double, b)
    w2tuple (Weight _avgloss _hits action) = (1, action)

-- | online mean accumulator.
updateAvgLoss :: Double -> Weight a -> Weight a
updateAvgLoss x w =
  w &~ do
    #hits += 1
    n <- use #hits <&> fromIntegral
    avg <- use #averageLoss
    #averageLoss += (x - avg) / (n + 1)

-- | updating the weights
updateWeight ::
  (Eq a) =>
  a ->
  Double ->
  NonEmpty (Weight a) ->
  NonEmpty (Weight a)
updateWeight a l = fmap updateIf
  where
    updateIf w@Weight {..} = if action == a then updateAvgLoss l w else w
