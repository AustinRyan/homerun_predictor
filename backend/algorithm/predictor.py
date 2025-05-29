import pandas as pd

# Placeholder for more sophisticated configuration or constants
MIN_PLATE_APPEARANCES_FOR_BATTER_STATS = 125 # Minimum PA for batter stats
MIN_BATTERS_FACED_FOR_PITCHER_STATS = 30 # Minimum batters faced for overall pitcher stats
DEFAULT_MIN_BF_PITCHER_SPLITS = 10 # Default min BF for pitcher handedness splits 
BVP_MIN_PA_FOR_SCORING = 5 # Minimum PAs for BvP stats to be considered in scoring # Minimum PA for a stat to be considered somewhat reliable
IDEAL_LAUNCH_ANGLE_MIN = 25.0
IDEAL_LAUNCH_ANGLE_MAX = 35.0
HARD_HIT_THRESHOLD = 95.0

# Default thresholds for handedness splits, FIP tiers, and BvP logic
DEFAULT_MIN_PA_HANDEDNESS_SPLITS = 5 # Default min PA for batter handedness splits
DEFAULT_STUD_PITCHER_FIP_THRESHOLD = 2.99
DEFAULT_BAD_PITCHER_FIP_THRESHOLD = 4.75
DEFAULT_STUD_PITCHER_FIP_ADJUSTMENT = -0.25
DEFAULT_BAD_PITCHER_FIP_ADJUSTMENT = 0.25
DEFAULT_BVP_PA_SIGNIFICANCE_THRESHOLD = 10 # Min BvP PAs to consider BvP stats in generate_hr_likelihood_score
DEFAULT_BVP_PA_SCALING_UPPER_THRESHOLD = 50 # BvP PAs at which BvP impact is maxed out in generate_hr_likelihood_score

def get_plate_appearances(player_df, batter_id=None, pitcher_id=None):
    """Helper to count plate appearances from Statcast data."""
    # This is a simplified way to count PAs. 
    # A more accurate way would be to group by game_pk, inning, and batter/pitcher matchup.
    # For now, we count rows where 'events' is not null (signifying end of PA)
    # and filter by batter or pitcher if provided.
    df = player_df.copy()
    if batter_id:
        df = df[df['batter'] == batter_id]
    if pitcher_id:
        df = df[df['pitcher'] == pitcher_id]
    
    # Count non-null 'events' as a proxy for PAs
    # This isn't perfect as some non-PA events might exist, but good for a start.
    # Statcast 'events' usually marks the outcome of a PA.
    return df['events'].notna().sum()

def calculate_batter_handedness_stats(player_statcast_df, batter_id, pitcher_throws=None, min_pa_threshold=10):
    """
    Calculates batter stats specifically against left or right-handed pitchers.
    
    Args:
        player_statcast_df (pd.DataFrame): DataFrame containing Statcast events for the batter.
        batter_id (int): The MLBAM ID of the batter.
        pitcher_throws (str): 'L' for left-handed pitchers, 'R' for right-handed pitchers.
        min_pa_threshold (int, optional): Minimum plate appearances threshold. Defaults to 10.
        
    Returns:
        dict: A dictionary of calculated stats for the batter against specified pitcher handedness.
    """
    stats = {'batter_id': batter_id, 'pitcher_throws': pitcher_throws}
    
    if player_statcast_df.empty:
        stats['error'] = 'No Statcast data available for this batter.'
        return stats
        
    # Filter for the specific batter
    batter_df = player_statcast_df[player_statcast_df['batter'] == batter_id].copy()
    
    # Further filter by pitcher handedness if specified
    if pitcher_throws in ['L', 'R']:
        batter_df = batter_df[batter_df['p_throws'] == pitcher_throws]
    
    # Count plate appearances
    pa_count = get_plate_appearances(batter_df)
    stats['pa_count'] = pa_count
    
    if pa_count < min_pa_threshold:
        stats['error'] = f'Insufficient PAs ({pa_count}) against {pitcher_throws}-handed pitchers. Minimum required: {min_pa_threshold}.'
        return stats
    
    # Filter for batted ball events
    batted_balls = batter_df[batter_df['type'] == 'X'].copy()
    
    # Calculate home run rate
    hr_count = batter_df[batter_df['events'] == 'home_run'].shape[0]
    stats['home_run_rate_vs_handedness'] = hr_count / pa_count if pa_count > 0 else 0
    
    # Calculate ISO (Isolated Power)
    singles = batter_df[batter_df['events'] == 'single'].shape[0]
    doubles = batter_df[batter_df['events'] == 'double'].shape[0]
    triples = batter_df[batter_df['events'] == 'triple'].shape[0]
    at_bats = pa_count - batter_df[batter_df['events'].isin(['walk', 'hit_by_pitch', 'sac_bunt', 'sac_fly'])].shape[0]
    
    if at_bats > 0:
        slugging = (singles + (2 * doubles) + (3 * triples) + (4 * hr_count)) / at_bats
        batting_avg = (singles + doubles + triples + hr_count) / at_bats
        stats['iso_vs_handedness'] = slugging - batting_avg
    else:
        stats['iso_vs_handedness'] = 0
    
    # Derive 'barrel' column from 'launch_speed_angle' if available
    if 'launch_speed_angle' in batted_balls.columns:
        # Assuming 6 is the category for barrel from Statcast's launch_speed_angle
        batted_balls.loc[:, 'barrel'] = (batted_balls['launch_speed_angle'].fillna(-1) == 6).astype(int)
    
    # Calculate barrel percentage
    if 'barrel' in batted_balls.columns:
        # Only count batted ball events (type == 'X') to be consistent with overall stats
        batted_ball_events = batted_balls[batted_balls['type'] == 'X'].copy()
        total_batted_balls = len(batted_ball_events)
        
        if total_batted_balls > 0:
            barrel_count = batted_ball_events['barrel'].sum()
            stats['barrel_pct_vs_handedness'] = barrel_count / total_batted_balls
            print(f"DEBUG: {stats['pitcher_throws']}-handed barrel calculation - Barrel count: {barrel_count}, Total batted balls: {total_batted_balls}, Barrel %: {stats['barrel_pct_vs_handedness']:.3f}")
        else:
            stats['barrel_pct_vs_handedness'] = 0
    else:
        stats['barrel_pct_vs_handedness'] = 0
    
    return stats

def calculate_pitcher_handedness_stats(player_statcast_df, pitcher_id, batter_stands=None, min_bf_threshold=10):
    """
    Calculates pitcher stats specifically against left or right-handed batters.
    
    Args:
        player_statcast_df (pd.DataFrame): DataFrame containing Statcast events for the pitcher.
        pitcher_id (int): The MLBAM ID of the pitcher.
        batter_stands (str): 'L' for left-handed batters, 'R' for right-handed batters.
        min_bf_threshold (int, optional): Minimum batters faced threshold. Defaults to 10.
        
    Returns:
        dict: A dictionary of calculated stats for the pitcher against specified batter handedness.
    """
    stats = {'pitcher_id': pitcher_id, 'batter_stands': batter_stands}
    
    if player_statcast_df.empty:
        stats['error'] = 'No Statcast data available for this pitcher.'
        return stats
        
    # Filter for the specific pitcher
    pitcher_df = player_statcast_df[player_statcast_df['pitcher'] == pitcher_id].copy()
    
    # Further filter by batter handedness if specified
    if batter_stands in ['L', 'R']:
        pitcher_df = pitcher_df[pitcher_df['stand'] == batter_stands]
    
    # Count batters faced (similar to plate appearances)
    bf_count = get_plate_appearances(pitcher_df)
    stats['bf_count'] = bf_count
    
    if bf_count < min_bf_threshold:
        stats['error'] = f'Insufficient BF ({bf_count}) against {batter_stands}-handed batters. Minimum required: {min_bf_threshold}.'
        return stats
    
    # Filter for batted ball events
    batted_balls = pitcher_df[pitcher_df['type'] == 'X'].copy()
    
    # Calculate home run rate allowed
    hr_count = pitcher_df[pitcher_df['events'] == 'home_run'].shape[0]
    stats['home_run_rate_allowed_vs_handedness'] = hr_count / bf_count if bf_count > 0 else 0
    
    # Calculate barrel percentage allowed
    if 'barrel' in batted_balls.columns and len(batted_balls) > 0:
        barrel_count = batted_balls['barrel'].sum()
        stats['barrel_pct_allowed_vs_handedness'] = barrel_count / len(batted_balls)
    else:
        stats['barrel_pct_allowed_vs_handedness'] = 0
    
    return stats

def calculate_batter_overall_stats(player_statcast_df, batter_id, min_pa_threshold=MIN_PLATE_APPEARANCES_FOR_BATTER_STATS):
    """
    Calculates aggregated Statcast metrics for a given batter from a DataFrame 
    of their Statcast data over a period.

    Args:
        player_statcast_df (pd.DataFrame): DataFrame containing Statcast events for the batter.
        batter_id (int): The MLBAM ID of the batter.
        min_pa_threshold (int, optional): Minimum plate appearances threshold. Defaults to MIN_PLATE_APPEARANCES_FOR_BATTER_STATS.

    Returns:
        dict: A dictionary of calculated stats for the batter.
    """
    if player_statcast_df.empty:
        return {}

    # Filter for the specific batter, though the input df should ideally be pre-filtered
    df = player_statcast_df[player_statcast_df['batter'] == batter_id].copy()
    if df.empty:
        return {}

    # Ensure key columns are numeric and handle potential errors
    numeric_cols = ['launch_speed', 'launch_angle', 'estimated_woba_using_speedangle', 'hit_distance_sc']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where essential metrics for calculation are NaN (e.g., launch_speed for hard_hit_rate)
    batted_balls = df.dropna(subset=['launch_speed', 'launch_angle']).copy()

    plate_appearances = get_plate_appearances(df, batter_id=batter_id)
    # effective_min_pa is now directly the min_pa_threshold argument

    if plate_appearances < min_pa_threshold:
        return {'batter_id': batter_id, 'plate_appearances': plate_appearances, 'error': f'Insufficient plate appearances ({plate_appearances} < {min_pa_threshold})'}

    total_batted_balls = len(batted_balls)
    if total_batted_balls == 0:
        return {'batter_id': batter_id, 'plate_appearances': plate_appearances, 'error': 'No batted ball data'}

    # Count hit types for ISO calculation
    singles = (df['events'] == 'single').sum()
    doubles = (df['events'] == 'double').sum()
    triples = (df['events'] == 'triple').sum()
    home_runs = (df['events'] == 'home_run').sum()
    
    # Calculate batting average and slugging for ISO
    at_bats = plate_appearances - (df['events'] == 'walk').sum() - (df['events'] == 'hit_by_pitch').sum() - (df['events'] == 'sac_fly').sum() - (df['events'] == 'sac_bunt').sum()
    hits = singles + doubles + triples + home_runs
    total_bases = singles + (2 * doubles) + (3 * triples) + (4 * home_runs)
    
    # Calculate ISO (Isolated Power = SLG - AVG)
    batting_avg = hits / at_bats if at_bats > 0 else 0
    slugging = total_bases / at_bats if at_bats > 0 else 0
    iso = slugging - batting_avg

    # Derive 'barrel' column from 'launch_speed_angle' if available
    if 'launch_speed_angle' in batted_balls.columns:
        # Assuming 6 is the category for barrel from Statcast's launch_speed_angle
        batted_balls.loc[:, 'barrel'] = (batted_balls['launch_speed_angle'].fillna(-1) == 6).astype(int)
    
    # Calculate Barrel Percentage
    barrel_pct_val = None
        
    if 'barrel' in batted_balls.columns:
        # Use the same calculation method as in the handedness-specific functions
        # Only count batted balls (type == 'X') to be consistent
        batted_ball_events = batted_balls[batted_balls['type'] == 'X'].copy()
        total_batted_balls_for_barrel = len(batted_ball_events)
        
        if total_batted_balls_for_barrel > 0:
            barrel_count = batted_ball_events['barrel'].sum()
            barrel_pct_val = barrel_count / total_batted_balls_for_barrel
            print(f"DEBUG: Overall barrel calculation - Barrel count: {barrel_count}, Total batted balls: {total_batted_balls_for_barrel}, Barrel %: {barrel_pct_val:.3f}")
        else:
            barrel_pct_val = 0.0
    
    # Calculate Sweet Spot Percentage (launch angle 8-32 degrees)
    sweet_spot_pct_val = None
    if 'launch_angle' in batted_balls.columns and not batted_balls['launch_angle'].isnull().all():
        sweet_spot_hits = batted_balls[(batted_balls['launch_angle'] >= 8) & (batted_balls['launch_angle'] <= 32)].shape[0]
        total_batted_balls_for_la = batted_balls['launch_angle'].dropna().shape[0] # Count non-NA launch angles for sweet spot
        if total_batted_balls_for_la > 0:
            sweet_spot_pct_val = sweet_spot_hits / total_batted_balls_for_la
        else:
            sweet_spot_pct_val = 0.0
    
    # Calculate exit velocity consistency (standard deviation of exit velocity)
    exit_velo_consistency = None
    if 'launch_speed' in batted_balls.columns and len(batted_balls) > 5:  # Need minimum sample
        exit_velo_consistency = batted_balls['launch_speed'].std()
        
    stats = {
        'batter_id': batter_id,
        'plate_appearances': plate_appearances,
        'avg_launch_speed': batted_balls['launch_speed'].mean(),
        'max_launch_speed': batted_balls['launch_speed'].max(),
        'avg_launch_angle': batted_balls['launch_angle'].mean(),
        'ideal_launch_angle_rate': (batted_balls['launch_angle'].between(IDEAL_LAUNCH_ANGLE_MIN, IDEAL_LAUNCH_ANGLE_MAX)).mean(),
        # Calculate hard hit rate using only batted ball events and the correct threshold
        'hard_hit_rate': (batted_ball_events['launch_speed'] >= HARD_HIT_THRESHOLD).mean() if 'type' in batted_balls.columns else (batted_balls['launch_speed'] >= HARD_HIT_THRESHOLD).mean(),
        'barrel_proxy_xwoba_on_contact': batted_balls['estimated_woba_using_speedangle'].mean(), # xwOBAcon
        'avg_hit_distance': batted_balls['hit_distance_sc'].mean(),
        'fly_ball_rate': (batted_balls['bb_type'] == 'fly_ball').mean(),
        'line_drive_rate': (batted_balls['bb_type'] == 'line_drive').mean(),
        'ground_ball_rate': (batted_balls['bb_type'] == 'ground_ball').mean(),
        'popup_rate': (batted_balls['bb_type'] == 'popup').mean(),
        'home_run_rate_per_pa': home_runs / plate_appearances if plate_appearances > 0 else 0,
        'iso': iso,  # Added ISO (Isolated Power)
        'batting_avg': batting_avg,
        'slugging': slugging,
        'home_run_rate_per_batted_ball': (batted_balls['events'] == 'home_run').sum() / total_batted_balls if total_batted_balls > 0 else 0,
        'barrel_pct': barrel_pct_val if barrel_pct_val is not None else 0.0,
        'sweet_spot_pct': sweet_spot_pct_val if sweet_spot_pct_val is not None else 0.0,
        'exit_velo_consistency': exit_velo_consistency,  # Added exit velocity consistency (lower is better)
    }
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items() if pd.notnull(v)}

def calculate_pitcher_overall_stats(player_statcast_df, pitcher_id, min_bf_threshold=MIN_BATTERS_FACED_FOR_PITCHER_STATS):
    """
    Calculates aggregated Statcast metrics for a given pitcher from a DataFrame
    of their Statcast data over a period (representing balls hit against them).

    Args:
        player_statcast_df (pd.DataFrame): DataFrame containing Statcast events against the pitcher.
        pitcher_id (int): The MLBAM ID of the pitcher.
        min_bf_threshold (int, optional): Minimum batters faced threshold. Defaults to MIN_BATTERS_FACED_FOR_PITCHER_STATS.

    Returns:
        dict: A dictionary of calculated stats for the pitcher.
    """
    if player_statcast_df.empty:
        return {}

    # Filter for the specific pitcher
    df = player_statcast_df[player_statcast_df['pitcher'] == pitcher_id].copy()
    if df.empty:
        return {}

    # Make sure key columns are numeric
    numeric_cols = ['launch_speed', 'launch_angle', 'estimated_woba_using_speedangle', 'hit_distance_sc']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Only look at batted balls with valid launch speed and angle for batted ball metrics
    batted_balls = df.dropna(subset=['launch_speed', 'launch_angle']).copy() if all(col in df.columns for col in ['launch_speed', 'launch_angle']) else pd.DataFrame()

    batters_faced = get_plate_appearances(df, pitcher_id=pitcher_id) # For pitchers, this counts batters faced
    
    # effective_min_bf is now directly the min_bf_threshold argument
    if batters_faced < min_bf_threshold:
        return {'pitcher_id': pitcher_id, 'batters_faced': batters_faced, 'error': f'Insufficient batters faced ({batters_faced} < {min_bf_threshold})'}

    total_batted_balls = len(batted_balls)
    if total_batted_balls == 0:
        return {'pitcher_id': pitcher_id, 'batters_faced': batters_faced, 'error': 'No batted ball data'}

    # Calculate components for FIP (Fielding Independent Pitching)
    hr_count = (df['events'] == 'home_run').sum()
    walks = (df['events'] == 'walk').sum() + (df['events'] == 'intent_walk').sum()
    hbp = (df['events'] == 'hit_by_pitch').sum() 
    k_count = (df['events'] == 'strikeout').sum()
    
    # Calculate innings pitched (estimate based on outs recorded)
    # This is a simplification; in a real implementation we'd track actual innings
    if 'outs_when_up' in df.columns:
        # If we have outs tracked in data
        innings = df['outs_when_up'].sum() / 3
    else:
        # Rough estimate based on events
        outs = (df['events'].isin(['field_out', 'strikeout', 'double_play', 'force_out', 'sac_fly', 'sac_bunt'])).sum()
        innings = outs / 3
        
    # FIP constant (approximately 3.10, this varies by year/league)
    fip_constant = 3.10
    
    # Calculate FIP if we have enough innings
    fip = None
    if innings > 0:
        fip = ((13 * hr_count) + (3 * (walks + hbp)) - (2 * k_count)) / innings + fip_constant

    # Derive 'barrel' column from 'launch_speed_angle' if available (for barrels allowed)
    if 'launch_speed_angle' in batted_balls.columns:
        # Assuming 6 is the category for barrel from Statcast's launch_speed_angle
        # Fill NA with -1 to ensure astype(int) doesn't fail and NA becomes 0 (not a barrel)
        batted_balls.loc[:, 'barrel'] = (batted_balls['launch_speed_angle'].fillna(-1) == 6).astype(int)

    # Calculate Barrel Percentage Allowed
    barrel_pct_allowed_val = None
    if 'barrel' in batted_balls.columns:
        total_batted_balls_for_barrel_allowed = len(batted_balls)
        if total_batted_balls_for_barrel_allowed > 0:
            barrel_allowed_count = batted_balls['barrel'].sum()
            barrel_pct_allowed_val = barrel_allowed_count / total_batted_balls_for_barrel_allowed
        else:
            barrel_pct_allowed_val = 0.0

    # Calculate Sweet Spot Percentage Allowed (launch angle 8-32 degrees)
    sweet_spot_pct_allowed_val = None
    if 'launch_angle' in batted_balls.columns and not batted_balls['launch_angle'].isnull().all():
        sweet_spot_hits_allowed = batted_balls[(batted_balls['launch_angle'] >= 8) & (batted_balls['launch_angle'] <= 32)].shape[0]
        total_batted_balls_for_la_allowed = batted_balls['launch_angle'].dropna().shape[0]
        if total_batted_balls_for_la_allowed > 0:
            sweet_spot_pct_allowed_val = sweet_spot_hits_allowed / total_batted_balls_for_la_allowed
        else:
            sweet_spot_pct_allowed_val = 0.0

    stats = {
        'pitcher_id': pitcher_id,
        'batters_faced': batters_faced,
        'avg_launch_speed_allowed': batted_balls['launch_speed'].mean(),
        'avg_launch_angle_allowed': batted_balls['launch_angle'].mean(),
        'ideal_launch_angle_rate_allowed': (batted_balls['launch_angle'].between(IDEAL_LAUNCH_ANGLE_MIN, IDEAL_LAUNCH_ANGLE_MAX)).mean(),
        'hard_hit_rate_allowed': (batted_balls['launch_speed'] >= HARD_HIT_THRESHOLD).mean(),
        'barrel_proxy_xwoba_on_contact_allowed': batted_balls['estimated_woba_using_speedangle'].mean(), # xwOBAcon
        'avg_hit_distance_allowed': batted_balls['hit_distance_sc'].mean(),
        'fly_ball_rate_allowed': (batted_balls['bb_type'] == 'fly_ball').mean(),
        'line_drive_rate_allowed': (batted_balls['bb_type'] == 'line_drive').mean(),
        'ground_ball_rate_allowed': (batted_balls['bb_type'] == 'ground_ball').mean(),
        'popup_rate_allowed': (batted_balls['bb_type'] == 'popup').mean(),
        'home_run_rate_allowed_per_pa': hr_count / batters_faced if batters_faced > 0 else 0.0,
        'home_run_rate_allowed_per_bf': (df['events'] == 'home_run').sum() / batters_faced if batters_faced > 0 else 0,
        'home_run_rate_allowed_per_batted_ball': (batted_balls['events'] == 'home_run').sum() / total_batted_balls if total_batted_balls > 0 else 0,
        'fip': fip,  # Added FIP (Fielding Independent Pitching)
        'strikeout_rate_per_bf': k_count / batters_faced if batters_faced > 0 else 0,
        'walk_rate_per_bf': walks / batters_faced if batters_faced > 0 else 0,
        'barrel_pct_allowed': barrel_pct_allowed_val if barrel_pct_allowed_val is not None else 0.0,
        'sweet_spot_pct_allowed': sweet_spot_pct_allowed_val if sweet_spot_pct_allowed_val is not None else 0.0,
        'innings_pitched': innings,
    }
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items() if pd.notnull(v)}

def calculate_batter_handedness_splits(statcast_df, batter_id):
    """
    Calculate batter performance vs LHP and RHP.
    
    Args:
        statcast_df (pd.DataFrame): DataFrame containing Statcast events.
        batter_id (int): The MLBAM ID of the batter.
        
    Returns:
        dict: A dictionary with vs_rhp and vs_lhp statistics and batter_stands info.
    """
    df = statcast_df[statcast_df['batter'] == batter_id].copy()
    
    # Store batter handedness
    batter_stands = df['stand'].iloc[0] if not df.empty and 'stand' in df.columns else None
    
    # Split data by pitcher handedness
    vs_rhp = df[df['p_throws'] == 'R']
    vs_lhp = df[df['p_throws'] == 'L']
    
    # Calculate metrics for each split with a lower PA threshold for splits
    rhp_stats = calculate_batter_overall_stats(vs_rhp, batter_id, min_pa_threshold=DEFAULT_MIN_PA_HANDEDNESS_SPLITS)
    lhp_stats = calculate_batter_overall_stats(vs_lhp, batter_id, min_pa_threshold=DEFAULT_MIN_PA_HANDEDNESS_SPLITS)
    
    return {
        'vs_rhp': rhp_stats,
        'vs_lhp': lhp_stats,
        'batter_stands': batter_stands 
    }

def calculate_pitcher_handedness_splits(statcast_df, pitcher_id):
    """
    Calculate pitcher performance vs LHB and RHB.
    
    Args:
        statcast_df (pd.DataFrame): DataFrame containing Statcast events.
        pitcher_id (int): The MLBAM ID of the pitcher.
        
    Returns:
        dict: A dictionary with vs_rhb and vs_lhb statistics and pitcher_throws info.
    """
    df = statcast_df[statcast_df['pitcher'] == pitcher_id].copy()
    
    # Store pitcher handedness
    pitcher_throws = df['p_throws'].iloc[0] if not df.empty and 'p_throws' in df.columns else None
    
    # Split data by batter handedness
    vs_rhb = df[df['stand'] == 'R']  # vs right-handed batters
    vs_lhb = df[df['stand'] == 'L']  # vs left-handed batters
    
    # Calculate metrics for each split
    rhb_stats = calculate_pitcher_overall_stats(vs_rhb, pitcher_id, min_bf_threshold=DEFAULT_MIN_BF_PITCHER_SPLITS)
    lhb_stats = calculate_pitcher_overall_stats(vs_lhb, pitcher_id, min_bf_threshold=DEFAULT_MIN_BF_PITCHER_SPLITS)
    
    return {
        'vs_rhb': rhb_stats,
        'vs_lhb': lhb_stats,
        'pitcher_throws': pitcher_throws
    }

def calculate_bvp_stats(statcast_df, batter_id, pitcher_id):
    """
    Calculates BvP (Batter vs. Pitcher) specific stats.
    This reuses the batter stats calculation logic but on a BvP-filtered DataFrame.

    Args:
        statcast_df (pd.DataFrame): DataFrame containing ALL Statcast events for a relevant period.
        batter_id (int): The MLBAM ID of the batter.
        pitcher_id (int): The MLBAM ID of the pitcher.

    Returns:
        dict: A dictionary of BvP stats.
    """
    bvp_df = statcast_df[(statcast_df['batter'] == batter_id) & (statcast_df['pitcher'] == pitcher_id)].copy()
    if bvp_df.empty:
        return {'batter_id': batter_id, 'pitcher_id': pitcher_id, 'error': 'No BvP history found'}
    
    bvp_pa = get_plate_appearances(bvp_df, batter_id=batter_id, pitcher_id=pitcher_id)
    if bvp_pa == 0:
        return {'batter_id': batter_id, 'pitcher_id': pitcher_id, 'bvp_plate_appearances': bvp_pa, 'error': 'No BvP plate appearances found'}
    
    # Initialize bvp_stats with metadata that's always relevant
    bvp_stats = {'batter_id': batter_id, 'pitcher_id_context': pitcher_id, 'bvp_plate_appearances': bvp_pa}

    if bvp_pa >= BVP_MIN_PA_FOR_SCORING:
        # Calculate BvP stats using the BVP_MIN_PA_FOR_SCORING as the threshold for actual calculation
        calculated_metrics = calculate_batter_overall_stats(bvp_df, batter_id, min_pa_threshold=BVP_MIN_PA_FOR_SCORING)
        bvp_stats.update(calculated_metrics) # Merge all calculated metrics (or error if calculation failed for other reasons)

        # If calculate_batter_overall_stats still returned an error (e.g., "No batted ball data"), it will be in bvp_stats.
        # Only add/override warning if no more severe error from the calculation itself.
        if 'error' not in bvp_stats:
            bvp_stats['warning'] = (f'BvP stats (based on {bvp_pa} PAs, min_pa_threshold={BVP_MIN_PA_FOR_SCORING}) will be used in score. '
                                    f'General min PAs for overall stats is {MIN_PLATE_APPEARANCES_FOR_BATTER_STATS}.')
        # If there was an error (like 'No batted ball data'), the scoring function will see it and skip BvP.

    else: # 0 < bvp_pa < BVP_MIN_PA_FOR_SCORING
        # PAs are too few for BvP stats to be reliably used in scoring.
        # We can still attempt to calculate them (which will likely result in an "insufficient PA" error from calculate_batter_overall_stats
        # due to its own internal check against min_pa_threshold which would be MIN_PLATE_APPEARANCES_FOR_BATTER_STATS by default).
        # Or, we can just add the error directly to signal they aren't used.
        calculated_metrics = calculate_batter_overall_stats(bvp_df, batter_id, min_pa_threshold=MIN_PLATE_APPEARANCES_FOR_BATTER_STATS)
        bvp_stats.update(calculated_metrics) # Merge, will likely include an error key from the PA check

        # Ensure a clear error/warning for the BvP context for scores/logs
        if 'error' not in bvp_stats:
             bvp_stats['error'] = f'BvP data has only {bvp_pa} PAs, less than BVP_MIN_PA_FOR_SCORING ({BVP_MIN_PA_FOR_SCORING}). Not used in score.'
        bvp_stats['warning'] = f'BvP stats based on very few PAs ({bvp_pa}). Not used in score. Error if present: {bvp_stats.get("error")}'
        
    #From the PA check
    if 'error' in bvp_stats:
        bvp_stats['warning'] = f'BvP stats based on very few PAs ({bvp_pa}). Not used in score. Error if present: {bvp_stats.get("error")}'

        # Ensure a clear error/warning for the BvP context for scores/logs
        if 'error' not in bvp_stats:
             bvp_stats['error'] = f'BvP data has only {bvp_pa} PAs, less than BVP_MIN_PA_FOR_SCORING ({BVP_MIN_PA_FOR_SCORING}). Not used in score.'
        bvp_stats['warning'] = f'BvP stats based on very few PAs ({bvp_pa}). Not used in score. Error if present: {bvp_stats.get("error")}'
        
    return bvp_stats

def generate_hr_likelihood_score(
    batter_metrics, 
    pitcher_metrics, 
    ballpark_factor=1.0, 
    weather_score=0,
    custom_weights=None
):
    """
    Generates a comprehensive score for home run likelihood using advanced metrics,
    handedness, and environmental factors.

    Args:
        batter_metrics (dict): Calculated stats for the batter.
        pitcher_metrics (dict): Calculated stats for the pitcher.
        ballpark_factor (float, optional): Park factor for home runs. Defaults to 1.0.
        weather_score (float, optional): Weather impact score (additive). Defaults to 0.

    Returns:
        float: A heuristic likelihood score, scaled 0-100.
    """
    # Log input metrics to verify fetched values vs defaults
    print(f"DEBUG_INPUT_METRICS: Batter ID: {batter_metrics.get('batter_id', 'N/A')}")
    batter_keys_to_log = ['avg_launch_speed', 'ideal_launch_angle_rate', 'hard_hit_rate', 'home_run_rate_per_pa', 'iso', 'barrel_pct', 'sweet_spot_pct']
    for key in batter_keys_to_log:
        value = batter_metrics.get(key, 'USING_DEFAULT')
        print(f"  Batter {key}: {value}")
    
    print(f"DEBUG_INPUT_METRICS: Pitcher ID: {pitcher_metrics.get('pitcher_id', 'N/A')}")
    pitcher_keys_to_log = ['avg_launch_speed_allowed', 'ideal_launch_angle_rate_allowed', 'hard_hit_rate_allowed', 'home_run_rate_allowed_per_pa', 'fip', 'barrel_pct_allowed', 'sweet_spot_pct_allowed', 'fly_ball_rate_allowed', 'ground_ball_rate_allowed']
    for key in pitcher_keys_to_log:
        value = pitcher_metrics.get(key, 'USING_DEFAULT')
        print(f"  Pitcher {key}: {value}")
    print(f"  Batter Stands: {batter_metrics.get('batter_stands', 'N/A')}, Pitcher Throws: {pitcher_metrics.get('pitcher_throws', 'N/A')}")
    # Debug: Print incoming environmental factors
    print(f"DEBUG_ENV_FACTORS: ballpark_factor={ballpark_factor}, weather_score={weather_score}")
    raw_score = 0.0
    
    defaults = {
        'avg_launch_speed': 88.0,
        'ideal_launch_angle_rate': 0.12,
        'hard_hit_rate': 0.35,
        # 'barrel_proxy_xwoba_on_contact': 0.350, # Removed, barrel_pct is primary
        'home_run_rate_per_pa': 0.03,
        'iso': 0.145,
        'barrel_pct': 0.06,
        'sweet_spot_pct': 0.33, # Typical sweet spot rate
        'exit_velo_consistency': 7.5, # Typical exit velo std dev
        'fip': 4.20,
        'fly_ball_rate_allowed': 0.35,
        'ground_ball_rate_allowed': 0.45
    }
    # Optimized weights based on backtesting results (2025-03-01 to 2025-05-28)
    default_weights = {
        'batter_avg_launch_speed': -0.2,      # Changed from 0.1 to -0.2 (negative impact)
        'batter_ideal_launch_angle_rate': 2.0, # Increased from 1.5 to 2.0
        'batter_hard_hit_rate': 2.0,         # Increased from 1.5 to 2.0
        # 'batter_barrel_proxy_xwoba_on_contact': 0.05, # Removed
        'batter_home_run_rate_per_pa': 10.0, # Kept the same
        'batter_iso': 10.0,                  # Significantly increased from 3.0 to 10.0
        'batter_barrel_pct': 10.0,           # Significantly increased from 6.0 to 10.0
        'batter_sweet_spot_pct': 0.5,        # Decreased from 1.0 to 0.5
        'batter_exit_velo_consistency': 0.5, # Changed from -0.2 to 0.5 (positive impact)
        
        # Pitcher weights: positive if higher metric value is WORSE for pitcher (increases batter HR score)
        # Negative if higher metric value is BETTER for pitcher (decreases batter HR score)
        'pitcher_avg_launch_speed_allowed': 0.1,      # Reduced from -0.5 to 0.1 (small positive impact)
        'pitcher_ideal_launch_angle_rate_allowed': 1.0, # Reduced from 2.0 to 1.0
        'pitcher_hard_hit_rate_allowed': 1.0,         # Reduced from 1.5 to 1.0
        # 'pitcher_barrel_proxy_xwoba_on_contact_allowed': 0.05, # Removed
        'pitcher_home_run_rate_allowed_per_pa': 8.0,  # Reduced from 10.0 to 8.0 (less weight on HR allowed)
        'pitcher_fip': 2.0,                           # Reduced from 3.0 to 2.0 (less weight on FIP)
        'pitcher_barrel_pct_allowed': 5.0,            # Significantly reduced from 10.0 to 5.0
        'pitcher_sweet_spot_pct_allowed': 0.3,        # Further decreased from 0.5 to 0.3
        'pitcher_fly_ball_rate_allowed': 0.3,         # Reduced from 0.5 to 0.3
        'pitcher_ground_ball_rate_allowed': -0.3,     # Adjusted from -0.2 to -0.3

        # Handedness-specific weights
        'handedness_advantage_bonus': 0.15, # Basic platoon advantage
        'batter_hr_rate_vs_handedness': 8.0, # Weight for batter's HR rate vs specific pitcher handedness
        'batter_iso_vs_handedness': 8.0,    # Weight for batter's ISO vs specific pitcher handedness
        'batter_barrel_pct_vs_handedness': 8.0, # Weight for batter's barrel% vs specific pitcher handedness
        'pitcher_hr_rate_allowed_vs_handedness': 6.0, # Weight for pitcher's HR rate allowed vs specific batter handedness
        'pitcher_barrel_pct_allowed_vs_handedness': 4.0 # Weight for pitcher's barrel% allowed vs specific batter handedness
    }
    
    # Use custom weights if provided, otherwise use defaults
    weights = custom_weights if custom_weights is not None else default_weights

    # Overrides removed, defaults and weights are now fixed as defined above

    # --- Batter Contributions ---
    if batter_metrics and not batter_metrics.get('error'):
        raw_score += (batter_metrics.get('avg_launch_speed', defaults['avg_launch_speed']) - defaults['avg_launch_speed']) * weights['batter_avg_launch_speed']
        raw_score += (batter_metrics.get('ideal_launch_angle_rate', defaults['ideal_launch_angle_rate']) - defaults['ideal_launch_angle_rate']) * weights['batter_ideal_launch_angle_rate']
        raw_score += (batter_metrics.get('hard_hit_rate', defaults['hard_hit_rate']) - defaults['hard_hit_rate']) * weights['batter_hard_hit_rate']
        # raw_score += (batter_metrics.get('barrel_proxy_xwoba_on_contact', defaults.get('barrel_proxy_xwoba_on_contact', 0.350)) - defaults.get('barrel_proxy_xwoba_on_contact', 0.350)) * weights.get('batter_barrel_proxy_xwoba_on_contact', 0) # Removed
        raw_score += (batter_metrics.get('home_run_rate_per_pa', defaults['home_run_rate_per_pa']) - defaults['home_run_rate_per_pa']) * weights['batter_home_run_rate_per_pa']
        if batter_metrics.get('iso') is not None:
            raw_score += (batter_metrics['iso'] - defaults['iso']) * weights['batter_iso']
        if batter_metrics.get('barrel_pct') is not None:
            raw_score += (batter_metrics['barrel_pct'] - defaults['barrel_pct']) * weights['batter_barrel_pct']
        if batter_metrics.get('sweet_spot_pct') is not None:
            raw_score += (batter_metrics['sweet_spot_pct'] - defaults['sweet_spot_pct']) * weights['batter_sweet_spot_pct']
        if batter_metrics.get('exit_velo_consistency') is not None:
            raw_score += (batter_metrics['exit_velo_consistency'] - defaults['exit_velo_consistency']) * weights['batter_exit_velo_consistency']

    # --- Pitcher Contributions ---
    if pitcher_metrics and not pitcher_metrics.get('error'):
        raw_score += (pitcher_metrics.get('avg_launch_speed_allowed', defaults['avg_launch_speed']) - defaults['avg_launch_speed']) * weights['pitcher_avg_launch_speed_allowed']
        raw_score += (pitcher_metrics.get('ideal_launch_angle_rate_allowed', defaults['ideal_launch_angle_rate']) - defaults['ideal_launch_angle_rate']) * weights['pitcher_ideal_launch_angle_rate_allowed']
        raw_score += (pitcher_metrics.get('hard_hit_rate_allowed', defaults['hard_hit_rate']) - defaults['hard_hit_rate']) * weights['pitcher_hard_hit_rate_allowed']
        raw_score += (pitcher_metrics.get('home_run_rate_allowed_per_pa', defaults['home_run_rate_per_pa']) - defaults['home_run_rate_per_pa']) * weights['pitcher_home_run_rate_allowed_per_pa']
        if pitcher_metrics.get('fip') is not None:
            raw_score += (pitcher_metrics['fip'] - defaults['fip']) * weights['pitcher_fip'] 
        if pitcher_metrics.get('barrel_pct_allowed') is not None:
            raw_score += (pitcher_metrics['barrel_pct_allowed'] - defaults['barrel_pct']) * weights['pitcher_barrel_pct_allowed']
        if pitcher_metrics.get('sweet_spot_pct_allowed') is not None:
            raw_score += (pitcher_metrics['sweet_spot_pct_allowed'] - defaults['sweet_spot_pct']) * weights['pitcher_sweet_spot_pct_allowed']
        if pitcher_metrics.get('fly_ball_rate_allowed') is not None:
            raw_score += (pitcher_metrics['fly_ball_rate_allowed'] - defaults['fly_ball_rate_allowed']) * weights['pitcher_fly_ball_rate_allowed'] 
        if pitcher_metrics.get('ground_ball_rate_allowed') is not None:
            raw_score += (pitcher_metrics['ground_ball_rate_allowed'] - defaults['ground_ball_rate_allowed']) * weights['pitcher_ground_ball_rate_allowed'] 

    # --- Handedness Advantage and Specific Performance ---
    current_batter_stands = batter_metrics.get('batter_stands') if batter_metrics else None
    current_pitcher_throws = pitcher_metrics.get('pitcher_throws') if pitcher_metrics else None

    # Basic platoon advantage (L vs R or R vs L)
    if current_batter_stands and current_pitcher_throws:
        if (current_batter_stands == 'L' and current_pitcher_throws == 'R') or \
           (current_batter_stands == 'R' and current_pitcher_throws == 'L'):
            raw_score += weights['handedness_advantage_bonus']
    
    # Batter's specific performance against this pitcher handedness
    if current_pitcher_throws and batter_metrics:
        # Home run rate vs this handedness
        if batter_metrics.get('home_run_rate_vs_handedness') is not None:
            # Compare to league average HR rate (approx. 0.035)
            raw_score += (batter_metrics['home_run_rate_vs_handedness'] - 0.035) * weights['batter_hr_rate_vs_handedness']
            
        # ISO vs this handedness
        if batter_metrics.get('iso_vs_handedness') is not None:
            # Compare to league average ISO (approx. 0.140)
            raw_score += (batter_metrics['iso_vs_handedness'] - 0.140) * weights['batter_iso_vs_handedness']
            
        # Barrel% vs this handedness
        if batter_metrics.get('barrel_pct_vs_handedness') is not None:
            # Compare to league average barrel% (approx. 0.08)
            raw_score += (batter_metrics['barrel_pct_vs_handedness'] - 0.08) * weights['batter_barrel_pct_vs_handedness']
    
    # Pitcher's specific performance against this batter handedness
    if current_batter_stands and pitcher_metrics:
        # Home run rate allowed vs this handedness
        if pitcher_metrics.get('home_run_rate_allowed_vs_handedness') is not None:
            # Compare to league average HR rate allowed (approx. 0.035)
            raw_score += (pitcher_metrics['home_run_rate_allowed_vs_handedness'] - 0.035) * weights['pitcher_hr_rate_allowed_vs_handedness']
            
        # Barrel% allowed vs this handedness
        if pitcher_metrics.get('barrel_pct_allowed_vs_handedness') is not None:
            # Compare to league average barrel% allowed (approx. 0.08)
            raw_score += (pitcher_metrics['barrel_pct_allowed_vs_handedness'] - 0.08) * weights['pitcher_barrel_pct_allowed_vs_handedness']

    # --- Score Finalization (Scaling and Environmental Factors) ---
    # The raw_score is an accumulation of positive/negative contributions.
    # A common scaling approach is logistic function or simple linear scaling.
    # Here, we'll use a linear scaling to a 0-100 range, centered around 50.
    # The sensitivity_factor determines how much raw_score changes affect the 0-100 scale.
    # With our optimized weights, we need to reduce this to prevent too many scores hitting 100
    # Original value was 10, but with higher weights we need a lower value
    sensitivity_factor = 5 
    scaled_score = 50 + (raw_score * sensitivity_factor)

    # Apply ballpark factor (multiplicative on the 0-100 scale, centered at 1.0)
    # e.g., if scaled_score is 60 and ballpark_factor is 1.1, it becomes 60 * 1.1 = 66
    # This assumes ballpark_factor is relative to an average park (1.0)
    # To make it relative to the midpoint (50) of the 0-100 scale:
    # final_score = 50 + (scaled_score - 50) * ballpark_factor
    # A simpler approach is to apply it to the raw score contributions before scaling, or adjust scaled score directly.
    # For now, apply as a multiplier to the scaled score, then re-center if needed or just scale.
    # Let's adjust the raw score by ballpark_factor first, then weather, then scale.
    # Assume ballpark_factor is a multiplier on the 'chance' of HR. 
    # If raw_score represents 'log-odds' or similar, this is complex. 
    # Simpler: raw_score is adjusted, then weather, then scale.
    
    # Adjusted raw score with ballpark and weather
    # Ballpark factor: if > 1, increases score; if < 1, decreases.
    # Let's assume raw_score is some form of 'advantage points'.
    # A ballpark_factor of 1.1 could mean a 10% increase in these points or on the probability.
    # For simplicity, let's make ballpark_factor additive to the raw_score based on its deviation from 1.
    # Example: ballpark_factor 1.1 adds (1.1-1.0)*X to raw_score. X is a sensitivity for ballpark.
    ballpark_raw_score_adjustment = (ballpark_factor - 1.0) * 2.0 # e.g. factor 1.1 adds 0.2 to raw_score
    raw_score_env_adjusted = raw_score + ballpark_raw_score_adjustment + weather_score
    
    final_scaled_score = 50 + (raw_score_env_adjusted * sensitivity_factor)
    
    # Clamp to 0-100 range
    final_scaled_score = max(0, min(100, final_scaled_score))
        
    # Logging for FIP tier and score components
    pitcher_id_log = pitcher_metrics.get('pitcher_id', 'N/A')
    pitcher_fip_log = pitcher_metrics.get('fip', None)
    
    fip_category_log = "Neutral"
    # Determine the actual fip_tier_adjustment that was applied to raw_score earlier in the function
    # This logic mirrors the adjustment application to ensure logged value is correct
    applied_fip_adjustment = 0
    if pitcher_fip_log is not None:
        if pitcher_fip_log <= DEFAULT_STUD_PITCHER_FIP_THRESHOLD:
            fip_category_log = "Stud"
            applied_fip_adjustment = DEFAULT_STUD_PITCHER_FIP_ADJUSTMENT
        elif pitcher_fip_log >= DEFAULT_BAD_PITCHER_FIP_THRESHOLD:
            fip_category_log = "Bad"
            applied_fip_adjustment = DEFAULT_BAD_PITCHER_FIP_ADJUSTMENT

    # 'raw_score' at this point in the function already includes the fip_tier_adjustment.
    # To log the score *before* fip_tier_adjustment, we would need to subtract it or log earlier.
    # For this log, 'raw_score' means 'score after component weights and fip tier adjustments, before env'.

    print(f"HR_SCORE_LOG: Pitcher ID: {pitcher_id_log}, FIP: {pitcher_fip_log if pitcher_fip_log is not None else 'N/A'}, \
          Tier: {fip_category_log}, Applied FIP Adj: {applied_fip_adjustment:.2f}, \
          Raw Score (post-FIP, pre-env): {raw_score:.2f}, Raw Score (post-env): {raw_score_env_adjusted:.2f}, \
          Final Score: {final_scaled_score:.2f}")

    return round(final_scaled_score, 2)


# Example Usage (conceptual - would be driven by main app logic)
if __name__ == '__main__':
    # This is a conceptual test. Actual data loading and filtering would be more complex.
    # print("Conceptual test for predictor.py - not runnable without data")
    
    # To make this runnable for a basic syntax check and conceptual validation:
    print("Running conceptual test for predictor.py...")

    # Create dummy DataFrames to simulate Statcast data
    dummy_batter_events = {
        'batter': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'pitcher': [10, 10, 11, 10, 11, 10, 10, 11, 10, 11, 10, 10, 11, 10, 11, 10, 10, 11, 10, 11, 10, 10, 11, 10, 11],
        'events': ['strikeout', 'field_out', 'single', 'home_run', 'walk', 'double', 'field_out', 'strikeout', 'single', 'home_run', 'field_out', 'single', 'home_run', 'walk', 'double', 'field_out', 'strikeout', 'single', 'home_run', 'field_out', 'single', 'home_run', 'walk', 'double', 'field_out'],
        'launch_speed': [None, 85.0, 95.0, 105.0, None, 100.0, 90.0, None, 92.0, 110.0, 85.0, 95.0, 105.0, None, 100.0, 90.0, None, 92.0, 110.0, 85.0, 95.0, 105.0, None, 100.0, 90.0],
        'launch_angle': [None, 10.0, 15.0, 30.0, None, 20.0, 5.0, None, 12.0, 28.0, 10.0, 15.0, 30.0, None, 20.0, 5.0, None, 12.0, 28.0, 10.0, 15.0, 30.0, None, 20.0, 5.0],
        'estimated_woba_using_speedangle': [None, 0.200, 0.600, 1.800, None, 0.900, 0.150, None, 0.500, 1.900, None, 0.200, 0.600, 1.800, None, 0.900, 0.150, None, 0.500, 1.900, None, 0.200, 0.600, 1.800, None],
        'hit_distance_sc': [None, 150, 250, 400, None, 300, 100, None, 220, 420, None, 150, 250, 400, None, 300, 100, None, 220, 420, None, 150, 250, 400, None],
        'bb_type': [None, 'ground_ball', 'line_drive', 'fly_ball', None, 'line_drive', 'ground_ball', None, 'line_drive', 'fly_ball', None, 'ground_ball', 'line_drive', 'fly_ball', None, 'line_drive', 'ground_ball', None, 'line_drive', 'fly_ball', None, 'ground_ball', 'line_drive', 'fly_ball', None]
    }
    dummy_pitcher_events = {
        'batter': [2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2],
        'pitcher': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        'events': ['field_out', 'strikeout', 'home_run', 'single', 'walk', 'strikeout', 'field_out', 'double', 'strikeout', 'home_run', 'field_out', 'strikeout', 'home_run', 'single', 'walk', 'strikeout', 'field_out', 'double', 'strikeout', 'home_run', 'field_out', 'strikeout', 'home_run', 'single', 'walk'],
        'launch_speed': [90.0, None, 100.0, 88.0, None, None, 92.0, 98.0, None, 108.0, 90.0, None, 100.0, 88.0, None, None, 92.0, 98.0, None, 108.0, 90.0, None, 100.0, 88.0, None],
        'launch_angle': [12.0, None, 35.0, 8.0, None, None, 15.0, 22.0, None, 25.0, 12.0, None, 35.0, 8.0, None, None, 15.0, 22.0, None, 25.0, 12.0, None, 35.0, 8.0, None],
        'estimated_woba_using_speedangle': [0.250, None, 1.500, 0.400, None, None, 0.300, 0.700, None, 1.700, 0.250, None, 1.500, 0.400, None, None, 0.300, 0.700, None, 1.700, 0.250, None, 1.500, 0.400, None],
        'hit_distance_sc': [180, None, 390, 150, None, None, 200, 280, None, 410, 180, None, 390, 150, None, None, 200, 280, None, 410, 180, None, 390, 150, None],
        'bb_type': ['ground_ball', None, 'fly_ball', 'ground_ball', None, None, 'line_drive', 'fly_ball', None, 'fly_ball', 'ground_ball', None, 'fly_ball', 'ground_ball', None, None, 'line_drive', 'fly_ball', None, 'fly_ball', 'ground_ball', None, 'fly_ball', 'ground_ball', None]
    }
    
    all_statcast_data = pd.concat([
        pd.DataFrame(dummy_batter_events),
        pd.DataFrame(dummy_pitcher_events)
    ]).reset_index(drop=True)
    # Add some BvP data for batter 1 vs pitcher 10
    # (already included in dummy_batter_events)

    example_batter_id = 1
    example_pitcher_id = 10

    # --- Calculate Batter Stats ---
    batter_overall_df = all_statcast_data[all_statcast_data['batter'] == example_batter_id]
    batter_stats = calculate_batter_overall_stats(batter_overall_df, example_batter_id)
    print(f"\nBatter Stats ({example_batter_id}):\n{pd.Series(batter_stats)}")

    # --- Calculate Pitcher Stats ---
    pitcher_overall_df = all_statcast_data[all_statcast_data['pitcher'] == example_pitcher_id]
    pitcher_stats_allowed = calculate_pitcher_overall_stats(pitcher_overall_df, example_pitcher_id)
    print(f"\nPitcher Stats ({example_pitcher_id} allowed):\n{pd.Series(pitcher_stats_allowed)}")
    
    # --- Calculate BvP Stats ---
    # For BvP, we use the full dataset and filter within the function
    bvp_specific_stats = calculate_bvp_stats(all_statcast_data, example_batter_id, example_pitcher_id)
    print(f"\nBvP Stats ({example_batter_id} vs {example_pitcher_id}):\n{pd.Series(bvp_specific_stats)}")

    # --- Generate Likelihood Score ---
    if batter_stats and 'error' not in batter_stats and \
       pitcher_stats_allowed and 'error' not in pitcher_stats_allowed:
        
        likelihood = generate_hr_likelihood_score(batter_stats, pitcher_stats_allowed, bvp_specific_stats)
        print(f"\nHome Run Likelihood Score for Batter {example_batter_id} vs Pitcher {example_pitcher_id}: {likelihood}")
    else:
        print("\nCould not generate likelihood score due to missing batter/pitcher overall stats or errors.")
        if 'error' in batter_stats: print(f"Batter error: {batter_stats['error']}")
        if 'error' in pitcher_stats_allowed: print(f"Pitcher error: {pitcher_stats_allowed['error']}")

    print("\nConceptual test finished.")
