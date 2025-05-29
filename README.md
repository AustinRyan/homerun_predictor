# Home Run Parlay Predictor

## Project Goal
To develop an algorithm and application that identifies statistically advantageous home run parlays for MLB games. The system will analyze various player, pitcher, game, and environmental metrics to provide data-driven insights.

## Tech Stack
- **Backend**: Python (Flask/FastAPI)
- **Frontend**: React with Vite
- **Data Sources**:
    - MLB Player/Game Stats: `statsapi.mlb.com` (requires registration)
    - Betting Odds: `The Odds API` (free tier available)
    - Weather: `Open-Meteo` (free, no API key for non-commercial)

## Directory Structure
homerun-model/
├── .gitignore
├── backend/
│   ├── app.py
│   └── requirements.txt
├── frontend/
│   └── (React/Vite project to be initialized here)
└── README.md

## Key Metrics to Consider
- **Batter Stats**:
    - Performance vs. RHP/LHP (AVG, SLG, OPS, ISO, HR rate)
    - Barrel Percentage
    - Hard Hit Percentage
    - Recent Performance (e.g., last 7, 15, 30 days)
    - Home/Away Splits
- **Pitcher Stats**:
    - HRs allowed (HR/9, HR/FB)
    - Performance vs. RHB/LHB
    - FIP (Fielding Independent Pitching)
    - Barrel Percentage Against
    - Hard Hit Percentage Against
    - Ground Ball / Fly Ball Ratio
    - Recent Performance
- **Game Factors**:
    - Ballpark Factors (e.g., ESPN Park Factors)
    - Weather Conditions (Temperature, Wind Speed/Direction, Humidity, Precipitation)
    - Umpire Tendencies (if data available)
- **Betting Odds**:
    - Implied probabilities from moneyline/total odds
    - Specific player prop odds for home runs

## Development Plan
1.  **Setup & Configuration**:
    *   [x] Initialize project structure (`.gitignore`, backend folders/files).
    *   [ ] Initialize React/Vite frontend.
    *   [ ] User to register for `statsapi.mlb.com`.
2.  **Backend Development**:
    *   [ ] Develop data fetching modules for MLB stats, odds, and weather.
    *   [ ] Implement data storage/caching strategy.
    *   [ ] Design and implement the core prediction algorithm.
    *   [ ] Create API endpoints (e.g., `/games`, `/players/{player_id}`, `/pitchers/{pitcher_id}`, `/predictions`).
3.  **Frontend Development**:
    *   [ ] Set up basic UI layout.
    *   [ ] Develop components for displaying games, player/pitcher stats.
    *   [ ] Implement functionality to interact with backend APIs.
    *   [ ] Design UI for presenting parlay suggestions and supporting data.
4.  **Algorithm Refinement & Testing**:
    *   [ ] Backtest algorithm against historical data.
    *   [ ] Continuously refine metrics and model based on performance.
5.  **Deployment**:
    *   [ ] Choose deployment platforms for backend and frontend.
    *   [ ] Deploy the application.

## Next Immediate Steps
1.  User to attempt registration for `statsapi.mlb.com`.
2.  Initialize the React/Vite frontend project.
3.  Begin development of the data fetching module for `Open-Meteo` (weather) in the backend.
