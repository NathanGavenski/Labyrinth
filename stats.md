# Structure

- metrics: dict[str, any]
    - train aer: float
    - train aer (str): float
    - train sr: float
    - eval aer: float
    - eval aer (str): float
    - eval sr: float
    - test aer: float
    - test aer (str): float
    - test sr: float
- solutions: List[str]
- non solutions: List[str]
- features: dict[str, any]
    - train
        - state (vector)
            - action: int
            - gt_action: int
            - features: np.array
    - eval
        - state (vector)
            - action: int
            - gt_action: int
            - features: np.array
    - test
        - state (vector)
            - action: int
            - gt_action: int
            - features: np.array


*sr*: success rate
*aer*: average episodic reward
