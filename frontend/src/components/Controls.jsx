import React from 'react';
import RefreshIcon from './icons/RefreshIcon';
import styles from './Controls.module.css';

const Controls = ({
  date,
  setDate,
  limit,
  setLimit,
  fetchPredictions
}) => {
  console.log("Controls component rendered - ID: " + Math.random().toString(36).substr(2, 5));

  return (
  <section className={styles.controlsSection}>
    <div className={styles.controlsGrid}>
      <div className={styles.controlItem}>
        <label htmlFor="date-input">Game Date</label>
        <input
          type="date"
          id="date-input"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          className={styles.inputField}
        />
      </div>
      <div className={styles.controlItem}>
        <label htmlFor="limit-input">Number of Predictions</label>
        <input
          type="number"
          id="limit-input"
          value={limit}
          onChange={(e) => setLimit(parseInt(e.target.value) || 5)}
          min="1"
          max="20"
          className={styles.inputField}
        />
      </div>
    </div>
    <div className={styles.controlActions}>
      <button onClick={() => {
          console.log("Calculate Predictions BUTTON CLICKED - ID: " + Math.random().toString(36).substr(2, 5));
          fetchPredictions();
        }} className={`${styles.actionButton} ${styles.refreshButton}`}>
        <RefreshIcon /> Calculate Predictions
      </button>
    </div>
  </section>
  );
};

export default Controls;
