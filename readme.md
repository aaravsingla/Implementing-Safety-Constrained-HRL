
## 1. Abstract
This project implements a **Contextual Multi-Armed Bandit (CMAB)** agent designed to optimize the timing of mobile health (mHealth) interventions. Addressing the challenge of "intervention fatigue," the system uses the **LinUCB (Linear Upper Confidence Bound)** algorithm to learn a user's latent preferences for notification receptivity based on contextual features (e.g., stress level, time of day, recent activity).

## 2. Motivation
In mobile health, the effectiveness of an intervention (e.g., a reminder to exercise or meditate) is heavily dependent on the context in which it is delivered. Static schedules often lead to habituation or dismissal.
* **The Goal:** Maximize user engagement (Click-Through Rate).
* **The Constraint:** Limit "bad" notifications to prevent user annoyance.
* **The Solution:** An adaptive agent that balances **Exploration** (trying new times to learn) and **Exploitation** (using the best known time).

## 3. Theoretical Framework
The simulation utilizes the **Disjoint LinUCB algorithm**. We assume the expected payoff (reward) of an arm $a$ is linear in its $d$-dimensional context feature vector $x_{t,a}$ with some unknown coefficient vector $\theta_a^*$.

The agent selects the action $a_t$ that maximizes the upper confidence bound estimate:

$$a_t = \underset{a \in \mathcal{A}_t}{\text{argmax}} \left( x_{t,a}^\top \hat{\theta}_a + \alpha \sqrt{x_{t,a}^\top \mathbf{A}_a^{-1} x_{t,a}} \right)$$

Where:
* $x_{t,a}^\top \hat{\theta}_a$: The estimated expected reward (Ridge Regression).
* $\alpha \sqrt{\dots}$: The exploration bonus (uncertainty width).
* $\mathbf{A}_a$: The covariance matrix ($d \times d$) accumulating context history.

## 4. Simulation Architecture
The project simulates a "Virtual Patient" environment to train and test the agent.

| Component | Description |
| :--- | :--- |
| **Context Space** | 5-dimensional vector representing user state (e.g., `[mood, stress, location, activity, fatigue]`). |
| **Action Space** | 4 discrete arms representing intervention windows (e.g., `Morning`, `Afternoon`, `Evening`, `No-Op`). |
| **Reward Signal** | Binary ($r \in \{0, 1\}$). Modeled stochastically using a sigmoid function over the dot product of context and true preference weights. |

## 5. Results
The agent demonstrates **sub-linear cumulative regret**, indicating successful convergence to the optimal policy.
* **Initial Phase:** High exploration (variance reduction).
* **Convergence:** The agent identifies the user's specific "receptivity windows" within ~500 interactions.

*(See the generic regret plots generated in the notebook for visual verification).*

## 6. Installation & Usage

### Prerequisites
* Python 3.x
* NumPy, Matplotlib, Pandas

### Running the Simulation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/adaptive-health-bandits.git](https://github.com/your-username/adaptive-health-bandits.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Launch the Jupyter Notebook:
    ```bash
    jupyter notebook contextual_bandits.ipynb
    ```

## 7. Future Work
* **Non-Stationarity:** Implementing a "Sliding Window LinUCB" to account for concept drift (e.g., user preferences changing on weekends vs. weekdays).
* **Risk Constraints:** Adding a penalty term to the reward function to strictly minimize interventions during high-stress contexts.

---
*This project was developed as a prototype for adaptive computational interaction in health systems.*