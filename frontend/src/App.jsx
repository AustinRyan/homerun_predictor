import { useState, useEffect } from 'react';
import styles from './App.module.css'; // Import CSS Module

// Import new components
import Header from './components/Header';
import Controls from './components/Controls';

import PredictionCard from './components/PredictionCard';
import LoadingSpinner from './components/LoadingSpinner';
import ErrorMessage from './components/ErrorMessage';
// Icons are now imported within their respective components (Header, Controls)

const initialWeights = {
  batter_avg_launch_speed: 0.1,
  batter_ideal_launch_angle_rate: 0.15,
  batter_hard_hit_rate: 0.15,
  batter_barrel_proxy_xwoba_on_contact: 0.1,
  batter_home_run_rate_per_pa: 0.1,
  pitcher_avg_launch_speed_allowed: -0.1,
  pitcher_ideal_launch_angle_rate_allowed: -0.15,
  pitcher_hard_hit_rate_allowed: -0.15,
  pitcher_barrel_proxy_xwoba_on_contact_allowed: -0.1,
  pitcher_home_run_rate_allowed_per_pa: -0.1,
  batter_iso: 0.1,
  batter_barrel_pct: 0.2,
  pitcher_fip: -0.1,
  pitcher_barrel_pct_allowed: -0.2,
  bvp_factor: 0.05,
  handedness_advantage: 0.05,
};

const initialDefaultValues = {
  avg_launch_speed: 88.0,
  ideal_launch_angle_rate: 0.12,
  hard_hit_rate: 0.35,
  barrel_proxy_xwoba_on_contact: 0.350,
  home_run_rate_per_pa: 0.03,
  iso: 0.145,
  barrel_pct: 0.06,
  fip: 4.20,
};


function App() {
  console.log("App component rendered - ID: " + Math.random().toString(36).substr(2, 5));
  const [date, setDate] = useState(new Date().toISOString().split('T')[0]);
  const [limit, setLimit] = useState(10); // Default to 10
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('cards'); // 'cards', 'table', or 'explanation'
  const [currentLoadingMessage, setCurrentLoadingMessage] = useState('');
  const [showStatsModal, setShowStatsModal] = useState(false);
  const [selectedPlayerStats, setSelectedPlayerStats] = useState(null);
  const [selectedPlayerType, setSelectedPlayerType] = useState('');

  const fetchPredictions = async () => {
    console.log("fetchPredictions called in App.jsx - ID: " + Math.random().toString(36).substr(2, 5));
    setLoading(true);
    setError(null);
    setPredictions([]);

    const apiUrl = `http://127.0.0.1:5000/api/homerun_predictions/${date}/${limit}`;
    const config = {
      limit: limit,
    };

    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ message: 'Network response was not ok.' }));
        throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      console.log('API Response Data:', data);
      console.log('First prediction sample:', data.predictions && data.predictions.length > 0 ? data.predictions[0] : 'No predictions');
      console.log('Available stats in first prediction:', data.predictions && data.predictions.length > 0 ? Object.keys(data.predictions[0]) : 'No predictions');
      console.log('Batter stats sample:', data.predictions && data.predictions.length > 0 ? data.predictions[0].batter_overall_stats : 'No stats');
      console.log('Pitcher stats sample:', data.predictions && data.predictions.length > 0 ? data.predictions[0].pitcher_overall_stats : 'No stats');
      console.log('Handedness adjusted batter stats:', data.predictions && data.predictions.length > 0 ? data.predictions[0].handedness_adjusted_batter_stats : 'No stats');
      setPredictions(data.predictions || []);
    } catch (err) {
      setError(err.message);
      console.error("Failed to fetch predictions:", err);
    } finally {
      setLoading(false);
    }
  };

  const loadingMessages = [
    'Initiating prediction sequence...',
    'Gathering game day schedules...',    
    'Fetching comprehensive Statcast data for the period...',
    'Processing player IDs and initial data cleaning...',    
    'Calculating overall batter performance metrics...',    
    'Assessing pitcher season statistics...',    
    'Analyzing batter handedness splits vs. pitchers...',    
    'Evaluating pitcher handedness splits vs. batters...',    
    'Incorporating ballpark-specific HR factors...',    
    'Checking for weather impacts (temperature, wind)...',    
    'Applying FIP-based pitcher quality adjustments...',    
    'Generating individual matchup likelihood scores...',    
    'Compiling and sorting final predictions...',    
    'Almost ready, just polishing the results!',
    'Finalizing predictions... Hang tight!'
  ];

  useEffect(() => {
    let intervalId = null;
    let messageIndex = 0;

    if (loading) {
      // Set the first message immediately
      setCurrentLoadingMessage(loadingMessages[messageIndex]);

      intervalId = setInterval(() => {
        messageIndex++;
        if (messageIndex < loadingMessages.length) {
          setCurrentLoadingMessage(loadingMessages[messageIndex]);
        } else {
          // All messages shown, clear interval to stop cycling and keep the last message.
          clearInterval(intervalId);
          // Ensure the last message is set, in case the interval clears slightly before the last update.
          setCurrentLoadingMessage(loadingMessages[loadingMessages.length - 1]);
        }
      }, 3000); // Change message every 3 seconds
    } else {
      setCurrentLoadingMessage(''); // Clear message when not loading
      if (intervalId) {
        clearInterval(intervalId); // Also clear interval if loading becomes false
      }
    }

    // Cleanup function
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [loading]);


  // Helper function to format percentage values (for table view)
  const formatPercent = (value) => {
    if (value === null || value === undefined) return 'N/A';
    return `${(value * 100).toFixed(1)}%`;
  };

  // Helper function to display handedness in a consistent way
  const getHandednessDisplay = (primaryHand, backupHand) => {
    const hand = primaryHand || backupHand;
    if (!hand) return '';  // Return empty string instead of showing parentheses
    return `(${hand})`;
  };

  // Helper function to determine score color class for table
  const getTableScoreClass = (score) => {
    if (score === null || score === undefined) return styles.tableScoreCellDefault;
    if (score >= 75) return styles.tableScoreCellHigh;
    if (score >= 50) return styles.tableScoreCellMedium;
    return styles.tableScoreCellLow;
  };

  // Function to display player stats in a modal when a player name is clicked
  const showPlayerStats = (overallStats, handednessStats, playerType, handedness) => {
    console.log('Overall stats:', overallStats);
    console.log('Handedness stats:', handednessStats);
    console.log('Player handedness:', handedness);
    console.log('Handedness stats keys:', handednessStats ? Object.keys(handednessStats) : 'No handedness stats');
    if (handednessStats) {
      if (handednessStats.vs_lhp) console.log('vs_lhp stats:', handednessStats.vs_lhp);
      if (handednessStats.vs_rhp) console.log('vs_rhp stats:', handednessStats.vs_rhp);
      if (handednessStats.vs_lhb) console.log('vs_lhb stats:', handednessStats.vs_lhb);
      if (handednessStats.vs_rhb) console.log('vs_rhb stats:', handednessStats.vs_rhb);
      if (handednessStats.home_run_rate_vs_handedness) console.log('HR rate vs handedness:', handednessStats.home_run_rate_vs_handedness);
    }
    
    // Try to determine handedness from multiple sources
    let effectiveHandedness = handedness;
    
    // If handedness is null, try to get it from the stats objects
    if (!effectiveHandedness) {
      if (playerType === 'Batter') {
        // Try to get batter handedness from various sources
        effectiveHandedness = overallStats?.batter_stands || 
                              handednessStats?.batter_stands || 
                              (handednessStats && 'vs_rhp' in handednessStats ? 'L' : null) || 
                              (handednessStats && 'vs_lhp' in handednessStats ? 'R' : null);
      } else {
        // Try to get pitcher handedness from various sources
        effectiveHandedness = overallStats?.pitcher_throws || 
                              handednessStats?.pitcher_throws || 
                              (handednessStats && 'vs_rhb' in handednessStats ? 'L' : null) || 
                              (handednessStats && 'vs_lhb' in handednessStats ? 'R' : null);
      }
    }
    
    // Add handedness to overall stats if it's not already there
    const updatedOverallStats = { ...overallStats };
    if (playerType === 'Batter') {
      updatedOverallStats.batter_stands = effectiveHandedness || 'Unknown';
    } else {
      updatedOverallStats.pitcher_throws = effectiveHandedness || 'Unknown';
    }
    
    setSelectedPlayerStats({
      overall: updatedOverallStats,
      handedness: handednessStats
    });
    setSelectedPlayerType(playerType);
    setShowStatsModal(true);
  };

  // Function to close the stats modal
  const closeStatsModal = () => {
    setShowStatsModal(false);
    setSelectedPlayerStats(null);
  };

  return (
    <div className={styles.appContainer}>
      <Header />
      <main className={styles.mainContent}>
        <Controls
          date={date}
          setDate={setDate}
          limit={limit}
          setLimit={setLimit}
          fetchPredictions={fetchPredictions}
        />

        <ErrorMessage message={error} />

        {loading && <LoadingSpinner message={currentLoadingMessage} />}

        <div className={styles.tabsContainer}>
          <div className={styles.tabs}>
            <button
              className={`${styles.tabButton} ${activeTab === 'cards' ? styles.activeTab : ''}`}
              onClick={() => setActiveTab('cards')}
            >
              Card View
            </button>
            <button
              className={`${styles.tabButton} ${activeTab === 'table' ? styles.activeTab : ''}`}
              onClick={() => setActiveTab('table')}
            >
              Table View
            </button>
            <button
              className={`${styles.tabButton} ${activeTab === 'explanation' ? styles.activeTab : ''}`}
              onClick={() => setActiveTab('explanation')}
            >
              How Scores Are Calculated
            </button>
          </div>
        </div>

        {activeTab === 'explanation' && (
          <section className={`${styles.resultsSection} ${styles.explanationSectionBackground}`}>
            <div className={styles.explanationContainer}>
              <h2>Understanding the Home Run Likelihood Score</h2>
              <p>
                The Home Run Likelihood Score is a comprehensive metric designed to predict the probability of a batter hitting a home run against a specific pitcher in a given game context. It integrates various statistical measures, historical data, and environmental factors. Here's a breakdown of how it's calculated:
              </p>

              <h3>1. Data Acquisition (<code>app.py</code>)</h3>    
              <ul>
                <li><strong>Statcast Data:</strong> The system fetches detailed Statcast data for the season up until the selected game date. This data includes pitch-by-pitch information, batted ball events, launch speed, launch angle, xwOBA (expected Weighted On-base Average), and more.</li>
                <li><strong>MLB Schedule:</strong> The daily MLB schedule is retrieved to identify the specific matchups (batter vs. pitcher) occurring on the selected date, along with venue information.</li>
                <li><strong>Weather Data (Future Implementation):</strong> While weather data fetching is part of the structure, its direct impact on scoring might be refined or expanded in future versions. Currently, temperature is a minor factor.</li>
              </ul>

              <h3>2. Core Statistical Calculations (<code>predictor.py</code>)</h3>
              <p>
                For each potential matchup, the backend calculates several layers of statistics. These functions use a minimum Plate Appearance (PA) threshold (e.g., 125 PAs for for batters, 50 batters faced for pitchers, 10 PAs for splits/BvP) to ensure statistical relevance. If a player is below this threshold, default league-average values are used for the missing stats.
              </p>
              <ul>
                <li>
                  <strong>Overall Player Stats (<code>calculate_batter_overall_stats</code>, <code>calculate_pitcher_overall_stats</code>):</strong>
                  Calculates a player's general performance metrics over the season up until today's date. Key stats include:
                  <ul>
                    <li>xwOBA (Expected Weighted On-base Average)</li>
                    <li>Barrel Percentage (Barrel%)</li>
                    <li>Hard Hit Percentage (HardHit%)</li>
                    <li>Home Run Rate per PA</li>
                    <li>Isolated Power (ISO)</li>
                    <li>For pitchers: FIP (Fielding Independent Pitching)</li>
                  </ul>
                </li>
                <li>
                  <strong>Handedness Splits (<code>calculate_batter_handedness_splits</code>, <code>calculate_pitcher_handedness_splits</code>):</strong>
                  Analyzes player performance specifically against left-handed or right-handed opponents. For example, a batter's xwOBA, Barrel%, and HardHit% against right-handed pitchers (RHP) and left-handed pitchers (LHP).
                </li>
                <li>
                  <strong>Batter vs. Pitcher (BvP) Stats (<code>calculate_bvp_stats</code>):</strong>
                  Examines the direct historical performance between the specific batter and pitcher in the matchup using the Statcast data. This looks at xwOBA, Barrel%, and HardHit% from their past encounters.
                </li>
              </ul>

              <h3>3. Generating the Likelihood Score (<code>generate_hr_likelihood_score</code> in <code>predictor.py</code>)</h3>
              <p>
                This is the core function where all the calculated stats are combined to produce the final score. The process involves several steps:
              </p>
              <ol>
                <li>
                  <strong>Metric Collection & Weighting:</strong>
                  The function gathers the relevant overall stats, handedness split stats, and BvP stats for both the batter and pitcher. Each statistic is assigned a specific weight, reflecting its perceived importance in predicting home runs. For example:
                  <ul>
                    <li>Batter's Overall Barrel%: +0.2</li>
                    <li>Pitcher's Overall Barrel% Allowed: -0.2 (negative because it's bad for the batter)</li>
                    <li>Batter's Handedness xwOBA: +0.075</li>
                    <li>BvP xwOBA: +0.15</li>
                    <li><em>(Note: Weights are configurable and can be adjusted in the backend.)</em></li>
                  </ul>
                  If a player lacks sufficient PAs for a specific stat (e.g., BvP data), a pre-defined default value (e.g., league average for that stat) is used instead. The difference between the player's stat and this default is then multiplied by the weight.
                </li>
                <li>
                  <strong>BvP Significance Scaling:</strong>
                  The impact of BvP stats is scaled based on the number of PAs in the BvP sample. A higher number of PAs (up to a scaling upper threshold, e.g., 50 PAs) gives the BvP data more influence. If BvP PAs are below a significance threshold, its contribution might be dampened or rely more on general/split stats.
                </li>
                <li>
                  <strong>Pitcher Quality Adjustment (FIP-based):</strong>
                  The pitcher's FIP (Fielding Independent Pitching) is used to adjust the score. 
                  <ul>
                      <li>'Stud' pitchers (e.g., FIP {'\u2264'} 3.50) incur a negative adjustment to the batter's HR likelihood (e.g., -0.25).</li>
                      <li>'Bad' pitchers (e.g., FIP {'\u2265'} 4.75) provide a positive adjustment (e.g., +0.25).</li>
                  </ul>
                  These adjustments and thresholds are also configurable.
                </li>
                <li>
                  <strong>Park Factor Adjustment:</strong>
                  Each MLB ballpark has a home run park factor (e.g., Coors Field is hitter-friendly, Oracle Park is pitcher-friendly). The score is adjusted based on the venue of the game. A factor greater than 1.0 increases HR likelihood, while less than 1.0 decreases it. The model uses a predefined dictionary of park factors.
                </li>
                <li>
                  <strong>Temperature Effect (Minor Factor):</strong>
                  A small adjustment is made based on game-time temperature, with warmer temperatures slightly increasing HR likelihood.
                </li>
                <li>
                  <strong>Raw Score Calculation:</strong>
                  The weighted contributions from all stats (overall, splits, BvP), FIP adjustments, park factor, and temperature are summed up to create a raw score.
                </li>
                <li>
                  <strong>Score Scaling & Normalization:</strong>
                  The raw score is then scaled to a 0-100 range to provide a more interpretable likelihood. This often involves a logistic function or a min-max scaling based on expected score ranges to ensure the final output represents a percentage-like likelihood.
                </li>
              </ol>

              <h3>4. Model Optimization & Validation</h3>
              <p>
                The weights used in our prediction model have been scientifically optimized using historical MLB data through a rigorous process:
              </p>
              <ol>
                <li>  
                  <strong>Weight Optimization:</strong>
                  I analyzed thousands of historical matchups from the 2025 MLB season, testing different weight combinations to maximize predictive accuracy. The optimization process used ROC-AUC (Receiver Operating Characteristic - Area Under Curve) as the primary metric, which measures how well the model distinguishes between home runs and non-home runs.
                </li>
                <li>
                  <strong>Backtesting Validation:</strong>
                  The optimized weights were validated through extensive backtesting across multiple months of MLB games. This process simulates how the model would have performed if used in real-time throughout the season. Our model achieved an overall ROC-AUC score of 0.591, indicating meaningful predictive power (scores above 0.5 show better-than-random prediction).
                </li>
                <li>
                  <strong>Key Findings:</strong>
                  The optimization revealed that barrel percentage and ISO (Isolated Power) are the most predictive metrics for home runs, with both receiving significantly higher weights in the final model.
                </li>
              </ol>

              <h3>5. Interpreting Scores</h3>
              <p>
                The final scores range from 0-100 and represent relative likelihood rather than absolute probability. Here's how to interpret them:
              </p>
              <ul>
                <li><strong>Scores 90+:</strong> Exceptionally high home run potential - these matchups are in the top tier of likelihood</li>
                <li><strong>Scores 80-89:</strong> Very strong home run potential - significantly above average likelihood</li>
                <li><strong>Scores 70-79:</strong> Strong home run potential - above average likelihood</li>
                <li><strong>Scores 60-69:</strong> Moderate home run potential - slightly above average likelihood</li>
                <li><strong>Scores 50-59:</strong> Average home run potential</li>
                <li><strong>Scores below 50:</strong> Below average home run potential</li>
              </ul>
              <p>
                <em>Note: Even matchups with high scores (90+) still have a relatively low absolute probability of resulting in a home run, as home runs are rare events (occurring in only about 7% of matchups in our backtesting data).</em>
              </p>

              <h3>6. Final Output (<code>app.py</code>)</h3>
              <p>
                The <code>app.py</code> endpoint takes all these calculated likelihood scores for the day's matchups, sorts them in descending order (highest likelihood first), and returns the top 'N' predictions (based on the user's limit) to the frontend for display.
              </p>
              <p>
                <em>This model is continuously refined. The specific weights, thresholds, and factors are subject to change as more data is analyzed and the prediction algorithm is improved.</em>
              </p>
            </div>
          </section>
        )}

        {!loading && !error && predictions.length > 0 && activeTab !== 'explanation' && (
          <section className={styles.resultsSection}>
            {activeTab === 'cards' && (
              <div className={styles.predictionCardsContainer}>
                {predictions.map((p, index) => (
                  <PredictionCard key={`${p.batter_id}-${p.pitcher_id}-${index}`} prediction={p} rank={index + 1} />
                ))}
              </div>
            )}

            {activeTab === 'table' && (
              <div className={styles.tableContainer}>
           
                <table className={styles.predictionsTable}>
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Batter</th>
                      <th>Pitcher</th>
                      <th>Score</th>
                      <th>Venue</th>
                      <th>Batter HR%</th>
                      <th>Batter ISO</th>
                      <th>Batter Barrel%</th>
                      <th>Pitcher HR%</th>
                      <th>Pitcher Barrel%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {predictions.map((p, index) => (
                      <tr key={p.matchup_id || index}>
                        <td>{index + 1}</td>
                        <td onClick={() => showPlayerStats(p.batter_overall_stats, p.handedness_adjusted_batter_stats, 'Batter', p.bat_hand)} className={styles.clickableCell}>{p.batter_name} {getHandednessDisplay(p.bat_hand, p.batter_overall_stats?.batter_stands)}</td>
                        <td onClick={() => showPlayerStats(p.pitcher_overall_stats, p.handedness_adjusted_pitcher_stats, 'Pitcher', p.pitch_hand)} className={styles.clickableCell}>{p.pitcher_name} {getHandednessDisplay(p.pitch_hand, p.pitcher_overall_stats?.pitcher_throws)}</td>
                        <td className={getTableScoreClass(p.hr_likelihood_score)}>{p.hr_likelihood_score?.toFixed(2) ?? 'N/A'}</td>
                        <td>{p.venue_name || 'N/A'}</td>
                        <td>{p.batter_overall_stats?.home_run_rate_per_pa ? (p.batter_overall_stats.home_run_rate_per_pa * 100).toFixed(1) + '%' : 'N/A'}</td>
                        <td>{p.batter_overall_stats?.iso ? p.batter_overall_stats.iso.toFixed(3) : 'N/A'}</td>
                        <td>{p.batter_overall_stats?.barrel_pct ? (p.batter_overall_stats.barrel_pct * 100).toFixed(1) + '%' : 'N/A'}</td>
                        <td>{p.pitcher_overall_stats?.home_run_rate_allowed_per_pa ? (p.pitcher_overall_stats.home_run_rate_allowed_per_pa * 100).toFixed(1) + '%' : 'N/A'}</td>
                        <td>{p.pitcher_overall_stats?.barrel_pct_allowed ? (p.pitcher_overall_stats.barrel_pct_allowed * 100).toFixed(1) + '%' : 'N/A'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}
        {!loading && !error && predictions.length === 0 && (
          <p className={styles.noPredictions}>No predictions available for the selected date or criteria. Try adjusting the settings.</p>
        )}
      </main>
      <footer className={styles.appFooter}>
        <p>MLB Home Run Predictor © {new Date().getFullYear()} | Data via PyBaseball & MLB Stats API</p>
      </footer>
      
      {/* Player Stats Modal */}
      {showStatsModal && selectedPlayerStats && (
        <div className={styles.modalOverlay} onClick={closeStatsModal}>
          <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h3 className={styles.modalTitle}>
                {selectedPlayerType} Statistics
              </h3>
              <button className={styles.closeButton} onClick={closeStatsModal}>×</button>
            </div>
            
            {/* Overall Stats Section */}
            <div className={styles.statSection}>
              <h4 className={styles.sectionTitle}>Overall Stats</h4>
              <div className={styles.statGrid}>
                {selectedPlayerType === 'Batter' ? (
                  <>
                    <div className={styles.statItem}>
                      <div className={styles.statLabel}>Handedness</div>
                      <div className={styles.statValue}>{selectedPlayerStats.overall?.batter_stands || selectedPlayerStats.overall?.pitcher_throws || 'N/A'}</div>
                    </div>
                    <div className={styles.statItem}>
                      <div className={styles.statLabel}>Home Run Rate</div>
                      <div className={styles.statValue}>
                        {selectedPlayerStats.overall?.home_run_rate_per_pa
                          ? `${(selectedPlayerStats.overall.home_run_rate_per_pa * 100).toFixed(2)}%`
                          : 'N/A'}
                      </div>
                    </div>
                    <div className={styles.statItem}>
                      <div className={styles.statLabel}>ISO (Isolated Power)</div>
                      <div className={styles.statValue}>
                        {selectedPlayerStats.overall?.iso
                          ? selectedPlayerStats.overall.iso.toFixed(3)
                          : 'N/A'}
                      </div>
                    </div>
                    <div className={styles.statItem}>
                      <div className={styles.statLabel}>Barrel %</div>
                      <div className={styles.statValue}>
                        {selectedPlayerStats.overall?.barrel_pct
                          ? `${(selectedPlayerStats.overall.barrel_pct * 100).toFixed(2)}%`
                          : 'N/A'}
                      </div>
                    </div>
                    {selectedPlayerStats.overall?.sweet_spot_pct && (
                      <div className={styles.statItem}>
                        <div className={styles.statLabel}>Sweet Spot %</div>
                        <div className={styles.statValue}>
                          {`${(selectedPlayerStats.overall.sweet_spot_pct * 100).toFixed(2)}%`}
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <>
                    <div className={styles.statItem}>
                      <div className={styles.statLabel}>Handedness</div>
                      <div className={styles.statValue}>{selectedPlayerStats.overall?.pitcher_throws || 'N/A'}</div>
                    </div>
                    <div className={styles.statItem}>
                      <div className={styles.statLabel}>HR Rate Allowed</div>
                      <div className={styles.statValue}>
                        {selectedPlayerStats.overall?.home_run_rate_allowed_per_pa
                          ? `${(selectedPlayerStats.overall.home_run_rate_allowed_per_pa * 100).toFixed(2)}%`
                          : 'N/A'}
                      </div>
                    </div>
                    <div className={styles.statItem}>
                      <div className={styles.statLabel}>Barrel % Allowed</div>
                      <div className={styles.statValue}>
                        {selectedPlayerStats.overall?.barrel_pct_allowed
                          ? `${(selectedPlayerStats.overall.barrel_pct_allowed * 100).toFixed(2)}%`
                          : 'N/A'}
                      </div>
                    </div>
                    {selectedPlayerStats.overall?.fip && (
                      <div className={styles.statItem}>
                        <div className={styles.statLabel}>FIP</div>
                        <div className={styles.statValue}>
                          {selectedPlayerStats.overall.fip.toFixed(2)}
                        </div>
                      </div>
                    )}
                    {selectedPlayerStats.overall?.sweet_spot_pct_allowed && (
                      <div className={styles.statItem}>
                        <div className={styles.statLabel}>Sweet Spot % Allowed</div>
                        <div className={styles.statValue}>
                          {`${(selectedPlayerStats.overall.sweet_spot_pct_allowed * 100).toFixed(2)}%`}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
            
            {/* Handedness-Specific Stats Section */}
            <div className={styles.statSection}>
              <h4 className={styles.sectionTitle}>
                {selectedPlayerType === 'Batter' 
                  ? `Stats vs ${selectedPlayerStats.overall?.batter_stands === 'L' ? 'RHP' : 'LHP'}`
                  : `Stats vs ${selectedPlayerStats.overall?.pitcher_throws === 'L' ? 'RHB' : 'LHB'}`}
              </h4>
              <div className={styles.statGrid}>
                {/* For batters, show handedness-specific stats */}
                {selectedPlayerType === 'Batter' && (
                  <>
                    {/* Check for home run rate vs handedness */}
                    {selectedPlayerStats.handedness?.home_run_rate_vs_handedness !== undefined ? (
                      <div className={styles.statItem}>
                        <div className={styles.statLabel}>HR Rate vs {selectedPlayerStats.overall?.batter_stands === 'L' ? 'RHP' : 'LHP'}</div>
                        <div className={styles.statValue}>
                          {`${(selectedPlayerStats.handedness.home_run_rate_vs_handedness * 100).toFixed(2)}%`}
                        </div>
                      </div>
                    ) : null}
                    
                    {/* Check for ISO vs handedness */}
                    {selectedPlayerStats.handedness?.iso_vs_handedness !== undefined ? (
                      <div className={styles.statItem}>
                        <div className={styles.statLabel}>ISO vs {selectedPlayerStats.overall?.batter_stands === 'L' ? 'RHP' : 'LHP'}</div>
                        <div className={styles.statValue}>
                          {selectedPlayerStats.handedness.iso_vs_handedness.toFixed(3)}
                        </div>
                      </div>
                    ) : null}
                    
                    {/* Check for barrel percentage vs handedness */}
                    {selectedPlayerStats.handedness?.barrel_pct_vs_handedness !== undefined ? (
                      <div className={styles.statItem}>
                        <div className={styles.statLabel}>Barrel % vs {selectedPlayerStats.overall?.batter_stands === 'L' ? 'RHP' : 'LHP'}</div>
                        <div className={styles.statValue}>
                          {`${(selectedPlayerStats.handedness.barrel_pct_vs_handedness * 100).toFixed(2)}%`}
                        </div>
                      </div>
                    ) : null}
                    
                    {/* Show N/A message if no handedness stats exist */}
                    {selectedPlayerStats.handedness?.home_run_rate_vs_handedness === undefined && 
                     selectedPlayerStats.handedness?.iso_vs_handedness === undefined && 
                     selectedPlayerStats.handedness?.barrel_pct_vs_handedness === undefined && (
                      <div className={styles.statItem} style={{ gridColumn: '1 / -1' }}>
                        <div className={styles.statLabel}>No handedness-specific stats available</div>
                        <div className={styles.statValue}>N/A</div>
                      </div>
                    )}
                  </>
                )}
                
                {/* For pitchers, show handedness-specific stats */}
                {selectedPlayerType === 'Pitcher' && (
                  <>
                    {/* Check for home run rate allowed vs handedness */}
                    {selectedPlayerStats.handedness?.home_run_rate_allowed_vs_handedness !== undefined ? (
                      <div className={styles.statItem}>
                        <div className={styles.statLabel}>HR Rate Allowed vs {selectedPlayerStats.overall?.pitcher_throws === 'L' ? 'RHB' : 'LHB'}</div>
                        <div className={styles.statValue}>
                          {`${(selectedPlayerStats.handedness.home_run_rate_allowed_vs_handedness * 100).toFixed(2)}%`}
                        </div>
                      </div>
                    ) : null}
                    
                    {/* Check for barrel percentage allowed vs handedness */}
                    {selectedPlayerStats.handedness?.barrel_pct_allowed_vs_handedness !== undefined ? (
                      <div className={styles.statItem}>
                        <div className={styles.statLabel}>Barrel % Allowed vs {selectedPlayerStats.overall?.pitcher_throws === 'L' ? 'RHB' : 'LHB'}</div>
                        <div className={styles.statValue}>
                          {`${(selectedPlayerStats.handedness.barrel_pct_allowed_vs_handedness * 100).toFixed(2)}%`}
                        </div>
                      </div>
                    ) : null}
                    
                    {/* Show N/A message if no handedness stats exist */}
                    {selectedPlayerStats.handedness?.home_run_rate_allowed_vs_handedness === undefined && 
                     selectedPlayerStats.handedness?.barrel_pct_allowed_vs_handedness === undefined && (
                      <div className={styles.statItem} style={{ gridColumn: '1 / -1' }}>
                        <div className={styles.statLabel}>No handedness-specific stats available</div>
                        <div className={styles.statValue}>N/A</div>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
