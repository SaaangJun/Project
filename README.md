# Project


Ideation

* Multi agent <-> Centralized Multi agent <-> Skill based RL 

<Skill based RL>
 1.매니저가 스킬을 보유한다. 
 
 2.이 스킬들은 하위 에이전트에게 전달된다.

  
 
 Env: 스위치 게임. 
       그리드 환경에 랜덤하게 에이전트, 스위치, 목표지점이 생긴다.
       스위치 두개를 밟고 있어야만 목표지점에 도착할 수 있다.
  
  매니저 보유 스킬:
  
  Skill 1. 각 에이전트들과 목표지점, 스위치들의 거리를 계산하여 제일 가까운 지점을 목표지점으로 설정한다.
  
  Skill 2. 상하좌우 이동  -----> Skill 1의 하위 액션들로 할 예정.
  
  Skill 3. 현재 자리에서 멈추기(Staying) -----> 1번과 통합할지 고민중.
