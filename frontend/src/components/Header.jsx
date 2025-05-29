import React from 'react';
import BaseballIcon from './icons/BaseballIcon';
import styles from './Header.module.css';

const Header = () => (
  <header className={styles.appHeader}>
    <div className={styles.appTitle}>
      <BaseballIcon /> MLB Home Run Predictor
    </div>
    <p className={styles.appSubtitle}>
      Advanced analytics to predict which batters are most likely to hit home runs based on matchups,
      ballpark factors, and historical performance.
    </p>
  </header>
);

export default Header;
