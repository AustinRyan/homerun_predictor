import React from 'react';
import styles from './CustomizeModel.module.css';

// Helper component for individual input fields to reduce repetition
const ParameterInput = ({ label, id, type = "number", step, value, onChange, ...props }) => (
  <div className={styles.controlItem}>
    <label htmlFor={id} className={styles.parameterLabel}>{label}</label>
    <input
      type={type}
      id={id}
      step={step}
      value={value}
      onChange={onChange}
      className={styles.parameterInput}
      {...props}
    />
  </div>
);

const CustomizeModel = ({
  // FIP Settings
  studFipThreshold, setStudFipThreshold,
  badFipThreshold, setBadFipThreshold,
  studFipAdjustment, setStudFipAdjustment,
  badFipAdjustment, setBadFipAdjustment,
  // Weight Params (shortened for brevity in props)
  weights, setWeight, // Pass a single weights object and a setter function
}) => {

  const fipParams = [
    { label: "Stud FIP Threshold", id: "stud-fip-threshold", value: studFipThreshold, action: setStudFipThreshold, step: "0.01" },
    { label: "Bad FIP Threshold", id: "bad-fip-threshold", value: badFipThreshold, action: setBadFipThreshold, step: "0.01" },
    { label: "Stud FIP Adjustment", id: "stud-fip-adjustment", value: studFipAdjustment, action: setStudFipAdjustment, step: "0.01" },
    { label: "Bad FIP Adjustment", id: "bad-fip-adjustment", value: badFipAdjustment, action: setBadFipAdjustment, step: "0.01" },
  ];

  // Define labels for weight parameters for easier rendering
  const weightLabels = {
    batter_avg_launch_speed: "Batter Avg Launch Speed",
    batter_ideal_launch_angle_rate: "Batter Ideal Launch Angle Rate",
    batter_hard_hit_rate: "Batter Hard Hit Rate",
    batter_barrel_proxy_xwoba_on_contact: "Batter Barrel Proxy xwOBA",
    batter_home_run_rate_per_pa: "Batter HR Rate / PA",
    pitcher_avg_launch_speed_allowed: "Pitcher Avg Launch Speed Allowed",
    pitcher_ideal_launch_angle_rate_allowed: "Pitcher Ideal Launch Angle Rate Allowed",
    pitcher_hard_hit_rate_allowed: "Pitcher Hard Hit Rate Allowed",
    pitcher_barrel_proxy_xwoba_on_contact_allowed: "Pitcher Barrel Proxy xwOBA Allowed",
    pitcher_home_run_rate_allowed_per_pa: "Pitcher HR Rate Allowed / PA",
    batter_iso: "Batter ISO",
    batter_barrel_pct: "Batter Barrel %",
    pitcher_fip: "Pitcher FIP",
    pitcher_barrel_pct_allowed: "Pitcher Barrel % Allowed",
    bvp_factor: "BvP Factor",
    handedness_advantage: "Handedness Advantage",
  };


  return (
    <section className={styles.customizeSection}>
      <h3 className={styles.sectionTitle}>Model Customization Parameters</h3>

      {/* FIP Settings Card */}
      <div className={styles.parameterCard}>
        <h4 className={styles.cardTitle}><span className={styles.icon}>⚙️</span> FIP Settings</h4>
        <div className={styles.parameterGrid}>
          {fipParams.map(param => (
            <ParameterInput
              key={param.id}
              label={param.label}
              id={param.id}
              value={param.value}
              onChange={(e) => param.action(parseFloat(e.target.value) || 0)}
              step={param.step}
            />
          ))}
        </div>
      </div>

      {/* Model Weight Parameters Card */}
      <div className={styles.parameterCard}>
        <h4 className={`${styles.cardTitle} ${styles.weightsTitle}`}><span className={styles.icon}>⚖️</span> Model Weight Parameters</h4>
        <div className={styles.parameterGrid}>
          {Object.entries(weights).map(([key, value]) => (
            <ParameterInput
              key={key}
              label={`W: ${weightLabels[key] || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}`}
              id={`weight_${key}`}
              value={value}
              onChange={(e) => setWeight(key, parseFloat(e.target.value) || 0)}
              step="0.01"
            />
          ))}
        </div>
      </div>
      {/* Note: Default value inputs are omitted for brevity but would follow a similar pattern if needed */}
    </section>
  );
};

export default CustomizeModel;
