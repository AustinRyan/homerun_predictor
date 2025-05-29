import React from 'react';
import styles from './LoadingSpinner.module.css';

const LoadingSpinner = ({ message }) => (
  <div className={styles.spinnerContainer}>
    <div className={styles.spinner}></div>
    <p className={styles.loadingText}>{message || 'Crunching the numbers... This may take a few minutes.'}</p>
  </div>
);

export default LoadingSpinner;
