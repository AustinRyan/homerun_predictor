import React from 'react';
import styles from './PredictionCard.module.css';

// Helper function to display handedness in a consistent way
const getHandednessDisplay = (primaryHand, backupHand) => {
  const hand = primaryHand || backupHand;
  if (!hand) return '';  // Return empty string instead of showing parentheses
  return `(${hand})`;
};

const PredictionCard = ({ prediction, rank }) => {
  const {
    batter_name,
    pitcher_name,
    venue_name,
    hr_likelihood_score,
    bat_hand,
    pitch_hand,
    home_team,
    away_team,
    batter_id,
    pitcher_id,
    batter_overall_stats,
    pitcher_overall_stats,
    // BvP data (optional)
    bvp_pa,
    bvp_hr,
    bvp_ba,
    bvp_slg,
    // ML Model data (optional)
    ml_probability,
    top_factors,
  } = prediction;

  const score = ml_probability !== undefined && ml_probability !== null 
                ? (ml_probability * 100).toFixed(1) 
                : hr_likelihood_score.toFixed(1);

  const scoreLabel = ml_probability !== undefined && ml_probability !== null 
                     ? "ML Prob." 
                     : "Likelihood";

  // Determine card border color based on score
  let borderColorClass = styles.borderNeutral;
  const numericScore = parseFloat(score);
  if (numericScore >= 75) borderColorClass = styles.borderHigh;
  else if (numericScore >= 50) borderColorClass = styles.borderMedium;
  else if (numericScore >= 25) borderColorClass = styles.borderLow;


  return (
    <div className={`${styles.card} ${borderColorClass}`}>
      <div className={styles.rankBadge}>#{rank}</div>
      <div className={styles.matchup}>
        <div className={styles.playerInfo}>
          <div className={styles.playerImageContainer}>
            <img 
              src={`https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/${batter_id}/headshot/67/current`} 
              alt={`${batter_name}`} 
              className={styles.playerImage} 
              onError={(e) => { e.target.src = 'https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/generic/headshot/67/current'; }}
            />
          </div>
          <span className={`${styles.playerName} ${styles.batterName}`}>{batter_name} {getHandednessDisplay(bat_hand, batter_overall_stats?.batter_stands)}</span>
        </div>
        <span className={styles.vs}>vs</span>
        <div className={styles.playerInfo}>
          <div className={styles.playerImageContainer}>
            <img 
              src={`https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/${pitcher_id}/headshot/67/current`} 
              alt={`${pitcher_name}`} 
              className={styles.playerImage} 
              onError={(e) => { e.target.src = 'https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/generic/headshot/67/current'; }}
            />
          </div>
          <span className={`${styles.playerName} ${styles.pitcherName}`}>{pitcher_name} {getHandednessDisplay(pitch_hand, pitcher_overall_stats?.pitcher_throws)}</span>
        </div>
      </div>
      <div className={styles.teams}>
        <span className={styles.teamName}>{away_team}</span>
        <span className={styles.teamSeparator}>@</span>
        <span className={styles.teamName}>{home_team}</span>
      </div>
      <div className={styles.venue}>🏟️ {venue_name}</div>

      <div className={styles.scoreContainer}>
        <div className={styles.scoreLabel}>{scoreLabel}</div>
        <div className={styles.scoreValue}>{score}</div>
      </div>

      {bvp_pa > 0 && (
        <div className={styles.bvpSection}>
          <h5 className={styles.bvpTitle}>Batter vs Pitcher ({bvp_pa} PA)</h5>
          <div className={styles.bvpStats}>
            <span>HR: {bvp_hr || 0}</span>
            <span>BA: {bvp_ba !== null && bvp_ba !== undefined ? bvp_ba.toFixed(3) : 'N/A'}</span>
            <span>SLG: {bvp_slg !== null && bvp_slg !== undefined ? bvp_slg.toFixed(3) : 'N/A'}</span>
          </div>
        </div>
      )}

      {top_factors && top_factors.length > 0 && (
        <div className={styles.factorsSection}>
          <h5 className={styles.factorsTitle}>Top Factors</h5>
          <ul className={styles.factorsList}>
            {top_factors.slice(0, 3).map((factor, index) => (
              <li key={index} className={styles.factorItem}>
                <span className={styles.factorName}>{factor.name.replace(/_/g, ' ')}:</span>
                <span className={styles.factorValue}>{factor.value.toFixed(2)} ({factor.contribution.toFixed(2)})</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default PredictionCard;
