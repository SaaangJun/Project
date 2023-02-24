# Project

어카농? 

걍 다른거로 만들어보는게 낫지 않나?

https://magent2.farama.org/environments/tiger_deer/

이거 괜찮아 보이는데 windows 기반으로도 돌아갈 수 있게 수정해야하지 않나
https://github.com/geek-ai/MAgent/blob/master/doc/get_started.md
https://github.com/Farama-Foundation/magent2

일단 다른 거 clone 해서 조금 가져다 써보는 걸 목적으로 해봐야 할 듯.

- 일단 matplotlib 이 아니라 gymnasium 을 상속해서 만들면 좋을 것 같은데
- 그 다음에 pygame 으로 visualization


---

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

  추가로 skill들과 action 이 정해지면 우선순위도 고려해야함.


***** 일단은 q-learning 기반 ( 추후 목적에 맞게 수정할 예정)  ***** 
 
 
To Do List:

1. Agent 추가

  - 이에 해당하는 action 정의
  
  - skill 추가 
  
  (어려운 점) : manager 생성 후 skill들을 코드에 어떤식으로 구성해야하는지


나중에 수정해야 할 부분:

   - 시작 점 랜덤 배치 (에이전트 개수 많아지면 코드 수정해야함)
 
 
Requirements:
python 3.7 이상
