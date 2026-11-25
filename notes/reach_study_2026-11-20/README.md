# 선반 D11 ↔ TurtleBot3 Burger 도달/경로 검토 (2026-11-20)

다음 세션 시작할 때: Claude에게 **"notes/reach_study_2026-11-20/README.md 읽고 이어서 하자"** 라고 하면 된다.

## 1. 목표 / 조건
- 트레이(16x10 cm 실물, 장면 모델 15x10x5.6 cm)를 선반 D11 ↔ 버거 박스 사이로 옮긴다. 손잡이를 위에서 수직으로 잡는다.
- **하드웨어 수정 불가.** 트레이를 **떨어뜨리면 안 됨**.
- 운반 중 기울기 ±10°는 허용 (잡기/놓기 순간은 수직).
- 버거 위치/방향은 매번 바뀔 수 있음 (yaw 고정 아님).

## 2. 현재 배치 (`~/.ros/layout.yaml`, 사본: `layout_at_end.yaml`)
- 책상: desk_x -1.0344..0.0656, edge_y -0.5062 (로봇이 +y 모서리에서 9.4 cm, +x 끝에서 6.6 cm), 책상 윗면 z -0.007, 바닥 z -0.306
- 선반: `shelf 0.36,0.11,90`, `shelf_x -0.097`, `fork_top 0.263`(최저), `fork_y 0`, `tray_yaw_off 90`
  → D11 윗면 (+0.003, +0.313, -0.043), r 31.3 cm. 책상/다리/받침/버거와 박스 겹침 없음.
- 버거(마지막 값): 박스 중심 (0.2456, -0.046), yaw 270.3 → 손잡이 z -0.208

## 3. 결론 요약
| 경로 | 결과 |
|---|---|
| D11, 정상 자세(J1≈+106°) | ✗ 위팔(link3)이 랙 앞 모서리(IR 기둥/왼쪽 패널, (-0.081,0.21,0.29))에 닿음. 로봇 J2가 옆으로 ~8.5 cm 오프셋이라 팔이 -x(랙 쪽)로 지나감 |
| D11, 반대 자세(J1≈-107°) | ✓ 잡기/접근, 선반 여유 3 cm까지 OK |
| D11 → 버거 직접 (트레이 든 채, 수직 / ±10°) | ✗ 길 없음: 가까이 돌면 트레이가 로봇 base, 멀리 돌면 마스트, 사이는 반대 자세 IK 구멍(방위~45°, r 16~19 cm) |
| **D11 → 책상 중간 자리 (반대 자세)** | ✓ 잡기 → 들기 → +x 2 cm → y 0.16까지 빼기 → z 0.125로 올림 → **트레이 제자리 +90° 회전** → ±10° 기울여 책상 위 자리로 → 세워서 내려놓기 (트레이 바닥이 책상 박스 여유 1.2 cm에 닿는 것은 정상 접촉) |
| 책상 자리 → 버거 위 z 0.125 (정상 자세, 트레이 +90° 그대로, ±10°) | ✓ 책상 자리 6곳 모두 경로 있음 |
| 버거 위에서 박스 안으로 내려놓기 | **✗ (마지막 결과, 원인 미확인)**: 버거 손잡이 (0.2456,-0.046), 트레이 yaw 90. z 0.125에서 수직 불가 → 기울인 채 내려가는 건 됨, 그런데 +8/+5/+2 cm 어디서도 "세우기(10°→0° 1°씩)" 실패. 다음: ① `fastik.solve`로 그 점에 수직 해가 있는지 (충돌 무시) ② 있으면 세우기 단계의 12° 점프 판정 문제인지 (단계 더 잘게) ③ 없으면 버거가 너무 멂(r 25.2) → 버거를 r 22 정도로 (이전 위치 (0.2155,-0.045)에서는 정상 자세 수직 +4/+8 OK였음) |
| 책상 위에서 빈손 자세 전환 (반대 ↔ 정상, J1 ~200°) | **미확인**: 책상 위 트레이를 장애물로 두고 MoveIt 경로 계획 필요 |
| 반대 방향 버거 → 책상 → D11 | 같은 경로의 역순이라 위가 다 되면 성립 (자세 전환 확인은 동일하게 필요) |

책상 중간 자리 후보(트레이가 책상 위에 다 올라가고 반대 자세로 도달): (-0.16,-0.14), (-0.14,-0.16), (-0.12,-0.18), (-0.18,-0.12), (-0.20,-0.08), (-0.10,-0.20) 등 r 21~23 cm, 로봇 뒤쪽-오른쪽(-x,-y).

검토했지만 안 된 배치 (재검토 불필요):
- 선반 yaw 80~105°, +x로 이동: 정상 자세 불가
- 선반 좌우 반전 yaw -90(±15°), shelf_x +0.097: 잡기는 여유 1 cm뿐, 운반 경로 충돌
- 선반 ↔ 버거 자리 교환 (선반 +x 끝, 버거 +y): 버거 쪽에서 link3/link4가 책상 모서리에 닿음
- 로봇 +y 이동: 받침판 때문에 최대 ~1.7 cm, 효과 미미

## 4. 다음에 할 일 (순서대로)
1. `legb2.py` 다시 실행 → 버거 박스에 내려놓기(기울여 내려간 뒤 세우기) 확인
2. 책상 위 빈손 자세 전환 경로: 책상 위 트레이를 collision object로 넣고 반대→정상, 정상→반대 둘 다 OMPL 계획 + 3 cm 여유로 서브스텝 검사 (`pathchk.py` 방식)
3. 버거 yaw 범위(예: 현재 값 ±30°)로 2번째 구간 반복
4. 다 되면 `tray_handle_pick`에 "책상 경유(relay)" 흐름 추가 제안 → **사용자 확인 후 코드 수정**
5. 안전 수정 제안: `tray_handle_pick` 계획기가 운반 구간에 **트레이를 충돌 모델에 넣지 않음** (그리퍼 0.2만 반영). 트레이 attach 추가 필요 → 사용자 확인 후

### ⚠ 먼저 확인할 것: 트레이 방향 모델
`held.py`는 트레이 15 cm를 **TCP x(손가락이 닫히는 방향)**로 붙였다 (place_scene의 선반 트레이 모델: 손잡이 바 91 mm가 트레이 폭 방향).
그런데 메모리의 실물 정보는 "트레이 16x10, **손잡이가 16 cm 변을 따라**" → 손가락은 10 cm 방향으로 닫히고 트레이 긴 쪽은 **TCP y**일 수 있다.
실물이 그렇다면 `held.py`의 box dimensions를 `[0.10, 0.16, 0.056]`(실물 크기)로 바꾸고 3장의 표를 다시 계산해야 한다. 그러면 결과가 바뀔 수 있다.

## 5. 코드 변경 상태 (커밋 안 함)
- `mycobot_tasks/layout.py`(untracked): DEFAULTS를 위 배치(책상/버거/선반)로, `--above` 기본 0.04, `--reset`/RViz reset이 기본 버거 위치 사용, "go above ..." 메뉴에 반대 자세 fallback + `--no-flip`
- `handle_pick/plan.py`, `cli.py`: `ik_above_all`에 반대 자세 fallback(`_ik_above`), `--no-flip` 옵션
- **반대 자세 코드는 유지/되돌림을 사용자가 결과 보고 결정하기로 함** (아직 결정 안 됨)

## 6. 스크립트 (`scripts/`, 원본은 세션 스크래치 폴더에서 복사)
실행: 시뮬레이션 + layout 노드가 떠 있어야 함
```
ros2 launch mycobot_calibration handeye.launch.py sim:=true scene_args:="--layout"
ros2 run mycobot_tasks layout
cd ~/final_project_ws/notes/reach_study_2026-11-20/scripts
source /opt/ros/*/setup.bash; source ~/final_project_ws/install/setup.bash
python3 legb2.py 2>&1 | grep -v '^\[INFO'
```
- `fastik.py`: URDF FK + scipy 수치 IK (MoveIt IK보다 훨씬 빠름, 충돌 없음)
- `scanlib.py`: ROS 노드(멀티스레드 spin), `apply_shelf(margin)`로 선반 박스 임시 교체, `valid(q)` 충돌 검사
  - 주의: MoveItClient 쓰는 스크립트는 노드를 spin해야 함 (안 하면 IK가 전부 None — 초반 결과 무효였던 원인)
  - `check_state_validity`는 항상 base-desk 접촉을 보고함 (compute_ik는 무시) → 무시하고 있음
- `route.py`: `ik_local`(이전 해 근처), `ik_any(p, yaw, flip)` (flip=True 반대 자세), `contacts`
- `held.py`: 트레이(15x10x5.6)를 gripper_tcp에 attach + 그리퍼 0.2로 충돌 검사(`contacts_held`), `chain`(5 mm 간격 연속 IK, 12° 점프 = 자세 전환으로 거부)
- `tilt2.py`: ±10° 기울기 IK(`ik_tilt`) + 2 cm 격자 Dijkstra로 D11→책상 자리 탐색
- `legb.py` / `legb2.py`: 책상 자리 → 버거 (정상 자세) 구간
- `overlap0.py`, `swapscan.py`, `rotscan.py`, `mirror.py`: 배치 겹침/회전/반전 탐색
- 스크립트가 선반을 임시로 바꾸면 RViz 장면 박스만 잠깐 바뀌고 layout 마커는 안 움직임 (끝나면 복원)
- 계산 중엔 RViz 배치를 건드리지 말 것 (장면 기준으로 충돌 검사)

## 7. 숫자 참고
- 수직 그립 최대 r (방위 90°): fork_top 263 mm에서 손잡이+2 cm 33.9 / +8 cm 30.2 cm, 287 mm 32.7/27.9, 304 mm 31.6/25.9 (낮을수록 멀리 닿음)
- 버거 손잡이(z -0.208): r 22~24 cm 정상 자세 OK, 반대 자세도 +4/+8 cm OK
- 마스트 앞면 대략 x 0.11~0.22, y ≥ 0.23 / 랙 앞 x ≤ -0.08, y ≥ 0.20 (현재 배치 base 좌표), D11은 그 사이 ~17 cm 틈

## 8. 2026-11-23 진행
- 트레이 방향: 실제 선반 트레이 = place_scene 모델(15x10, 91 mm 바가 10 cm 방향). 16x10은 테스트 트레이일 뿐 → `held.py` 그대로 맞음.
- 버거 박스 내려놓기 실패 원인: 박스 위 수직 해는 있음(`scripts/diag_box.py`, 정상 자세 J1≈10°, +8~0 cm 충돌 없음).
  기울인 채 내려간 자세가 **J6 = +180°(URDF 한계 ±3.14)**에 붙어 있고, 수직 해는 J6 = -170°(=+190°)라 세우기 불가.
- 해결(한계 그대로): 책상 자리에서 **반대 방향(+180°)으로 다시 잡고**, 박스용 트레이 yaw **270**으로 +90° 돌림
  → 책상 자리 6곳 모두 "버거 위 경로 + +8 cm에서 세우기 + 박스 안으로 내리기" OK (`scripts/legb3.py`).
  (다시 잡기 0°는 box 90 = J6 한계, box 270 = 돌리기 -90° 실패)
- 사용자: J6 실물은 ±270°까지 괜찮다고 함 → 한계 늘리면 다시 잡기 방향 제약도 풀릴 가능성. 아직 URDF 미변경.
- 경로 A 전체를 sim에서 재생: `scripts/route_a.py [spot_x spot_y]` (계산, ~30 s → route_a.pkl) → `scripts/play_route.py [--speed 0.35] [--pause]` (sim 팔 + RViz 트레이 박스 "route_tray").
  자리 (-0.16,-0.14): 18 구간 모두 성공, 빈손 자세 전환(OMPL, 책상 트레이 장애물, 2° 간격 충돌 검사) 포함, J6 -101..107°.
  책상 내려놓기는 트레이 바닥이 z 0.062부터 책상 여유(1.2 cm)에 들어가서 그 아래는 충돌 검사 안 함 (정상 접촉).
- 주의: 지금은 layout.yaml + layout_tray_bot.yaml의 **고정 좌표**. 실제로는 버거 위치/yaw가 매번 다르고 오차 있음 → 3단계(버거 위치/yaw 범위 검사)와 relay 흐름에서 측정값으로 매번 계획해야 함.

## 9. 2026-11-23 tray_handle_pick 기능화 (n = D11→버거, j = 버거→D11, `handle_pick/relay.py`)
- sim: `tray_handle_pick --layout-poses --scene-args=--layout [--go]` (RViz 배치를 n/j 때마다 읽음, 카메라 없음). 실물: 버거 마커 카메라 측정 필수, D11은 카메라(안 보이면 layout 값).
- 물건 안 쏟아지게: 기울기 최소화(`--relay-tilt-weight`) → 들고 있는 구간 최대 2.3~2.9°, 부드러운 가감속(≤6 cm/s, ≤0.15 m/s²).
- 계획 후 전체 경로 1° 간격 재검사(`relay_verify`), 트레이 5 mm 키워 근접 구간 보고(`--relay-pad`).
- 속도: A* + 처음 닿은 책상 자리 사용 + 계획 캐시 → 106 s → 45 s. 단 가까운 자리 (-0.20,-0.08)이 골라져 트레이가 로봇 base에 5 mm 이내(44 자세).
- `--relay-plan-pad 0.005 --bot-yaw-offset 90`: 양방향 OK, 5 mm 이내는 책상 정상 접촉뿐, 계획 64~89 s.
  `--bot-yaw-offset 0`(15 cm 변이 박스 16 cm 쪽)은 버거 본체와 ~2 mm → 5 mm 여유로는 경로 없음.
- 배치 변형(구 코드): 버거 +x 3 cm / +y 3 cm OK, **버거 -x 3 cm(로봇 쪽) 실패**(박스로 내릴 때 link3-책상). 나머지 변형은 아직 안 돌림 (`scripts/variants.py`).

## 10. 2026-11-23 배치 캠페인 (`scripts/rank2.py`, 기준 layouts/verified_1120, --relay-plan-pad 0.005 --bot-yaw-offset 90)
양방향 OK + 버거 ±2 cm: bot_r28 4/4, base 4/4, shelf_x-3 4/4, bot_yaw240 3/3, bot_az-25 3/4.
n만 실패: bot_yaw180 (촘촘한 검사에서 트레이-base 실제 충돌 발견). 실패: bot_r22 (책상→버거), shelf_x+3 / shelf_y+2 / shelf_yaw85 / fork_287 / robot ±90 (모두 D11 쪽).
겹쳐서 제외: bot_az0, bot_yaw300, shelf_yaw95. → 버거 쪽은 여유 있음, 선반(D11) 쪽은 몇 cm만 달라도 실패.

## 11. 배치 조건과 한계 (2026-11-23 최종, 기준 layouts/verified_1120)
아래는 sim에서 양방향(n, j) 계획과 촘촘한 충돌 검사를 모두 통과한 범위입니다. 한 번에 한 가지만 바꿔서 시험했습니다.
- 꼭 줄 옵션: `--relay-plan-pad 0.005 --bot-yaw-offset 90`. offset이 0이면 버거 몸체와 2 mm밖에 안 떨어져서 길이 막힙니다.
- **선반 (가장 예민함)**
  - x: -3 ~ 0 cm는 OK, **+1 cm부터 실패**합니다 (D11↔책상 길 없음). 지금 위치가 한계이므로 실제로는 -x(로봇 쪽)로 1~2 cm 당겨 두세요.
  - y: 0 ~ +1 cm는 OK, +2 cm는 실패합니다. -1 cm는 선반 받침이 책상 다리에 겹칩니다.
  - yaw: 88 ~ 90°는 OK, 85°는 실패합니다. 92°부터는 받침이 책상 다리에 겹칩니다.
  - 포크 높이: **260 / 263 / 270 mm는 OK**, 255 / 245 / 235 / 287 mm는 실패입니다.
    - 255~245 mm: 트레이를 꺼낼 때 책상 모서리에 닿고, 더 들면 카메라 플랜지가 랙에 닿습니다.
    - 235 mm: 드는 단계에서 포크에 닿습니다.
    - 따라서 사용 범위는 260~270 mm입니다 (실제 선반은 263~270 mm를 커버함).
  - 이 결과로 D11 손잡이는 r 약 31~32 cm가 한계입니다.
- **버거 (여유 있음)**
  - 거리 r: 23 ~ 30 cm는 OK, 22 cm는 실패합니다 (박스로 내릴 때 link3가 책상에 닿음). r 23 cm는 계획에 180 s가 걸리고, r 30 cm는 기울기가 4.1°까지 커집니다. 권장은 r 25~28 cm입니다.
  - 박스 yaw: 240 / 255 / 270°는 OK입니다. 180°에서는 n만 실패하고, 300°는 책상과 겹칩니다.
  - 방위: -25° ~ 현재 위치(약 -11°)는 OK입니다. 0°는 책상과 겹칩니다.
  - ±2 cm 오차가 있어도 대부분 성공합니다 (base 4/4, r28 4/4, yaw240 3/3, az-25 3/4).
- 로봇을 ±90° 돌린 배치는 실패합니다 (D11 쪽).
- 실제 운용: 카메라로 잰 D11 위치가 layout과 1~2 cm 넘게 다르면, 특히 +x 쪽이면 계획이 실패할 수 있습니다. 실패해도 아무것도 움직이지 않은 상태에서 멈춥니다.

> 11-24 메모: 11장의 "꼭 줄 옵션"은 이제 기본값입니다 (`--relay-plan-pad` 0.005,0.003,0 단계적, `--bot-yaw-offset 90`).
> 11-23의 offset 90 "성공"은 회전 중간을 검사하기 전이라 실제로는 base에 닿는 경로였습니다 (12.1 참고).

## 12. 2026-11-24 진행 (커밋 407475a, d074203 + 커밋 안 한 수정)

### 12.1 동작 변경 (`handle_pick/`)
| 내용 | 옵션 (기본값) |
|---|---|
| **e / r: 책상 ↔ 버거만** (`--do deskbot / botdesk`). e는 책상 자리 위에서만 트레이를 찾음, r은 박스 안 트레이를 찾음 | `--relay-desk x,y,yaw`, `--relay-spots` (기본 auto: base에서 20~23 cm) |
| 트레이를 base에서 먼 곳(책상 자리 → 35 % → 70 % → 박스 위)에서 회전. 책상 자리가 base에서 18~20 cm라 15 cm 트레이가 회전 중 base를 쓸었음 | `--bot-yaw-offset 90` (선반 트레이 15 cm 변 → 박스 20 cm 쪽) |
| 그리퍼: 닫은 채 접근 → 열기, 놓고 위로 빠진 뒤 닫기, 책상 자세 전환은 닫은 채 | – |
| 끝나면 홈 + 그리퍼 닫기. 집기 전 실패(트레이 없음, 버거 못 찾음, 계획 실패, 거리 밖)는 ERROR 후 홈 | `--relay-end home / stay` |
| 여유 단계적 계획 (5 → 3 → 0 mm, 구간마다), 0 mm면 WARNING | `--relay-plan-pad 0.005,0.003,0` |
| 받침면 접촉 허용 2 cm. 버거 박스를 바닥(`bot_box`)과 벽(`bot_box_walls`)으로 나눔: 벽(바닥 위 4 mm 턱)은 항상 충돌 검사 | `--relay-touch 0.02` |
| 계획 시간: IK 해석적 Jacobian (계획의 90 %가 IK였음, 10 → 3 ms), 격자 탐색 한도 100점 후 전체, 같은 박스 실패 반복 안 함, 시간 제한 | `--relay-grid-budget 100`, `--relay-plan-timeout 120` |
| 버거 거리 미리 검사 (밖이면 계획 없이 ERROR, 몇 mm 옮기라고 표시) | `--bot-r-range 0.23,0.30` |
| 놓을 때 트레이를 295 mm 안쪽에 (박스 중심 295~310 mm면 로봇 쪽으로 최대 15 mm 옮겨 놓음 → r이 측정 오차가 있어도 집음). 박스 중앙에서 못 내려가면 5 / 10 mm 비켜 놓기(마지막 단계에서만) | `--bot-place-r 0.295`, `--bot-place-in 0.015`, `--relay-place-shift 0.005,0.01` |
| 빈손 OMPL 이동(시작 위치, 자세 전환, 홈)은 책상 위 4 cm 이상 (base 주변 20 cm 제외), 목표가 더 높으면 먼저 수직으로 최대 15 cm 올림 | `--relay-desk-clear 0.04`, `--relay-desk-clear-r 0.10`, `--relay-lift-max 0.15` |
| r: 마지막 "arm switch back" 삭제 (책상에 놓고 바로 홈) | – |
| 트레이를 든 구간 속도 0.25 rad/s (빈손 0.35) — **실물 효과 아직 안 봄** | `--relay-held-speed 0.25` |
| r / j: 박스 안 트레이를 찾은 뒤 버거를 장면에 넣고 카메라 28 cm로 다시 잼 (처음 36 cm는 마커 36 px라 기울기 9~16°로 나옴 → 다시 재면 50 px, 2.0°) | `--bot-close-view 0.28` |
| 트레이 0~5 (마커 10+2t / 11+2t) 자동 선택, 잘못된 트레이 확인 자리 `trays.TrayCheck` (지금은 모두 통과) | – |
| 트레이 프리셋: shelf = 높이 76 mm, 15×10×5.6, offset 90, 잡는 깊이 24 / 꺼내는 높이 36 mm; test = 72 mm, 10×16×5.6(몸체 높이 미확인), offset 0, 20 / 40 mm | `--tray-type shelf / test` |
| 장면 / 배치: `place_scene --rack-trays [--rack-tray-out]`, layout 노드 `--scene-args` (launch가 전달), RViz 책상·선반 잠금, D11 위 트레이 핸들 (±5 cm, ±20°) | – |
| 드라이버: `~/final_project_ws/track_log.csv` (움직이는 중 계획−실제, 경로 이탈 off_norm, connection) | `track_log_csv` (driver.yaml) |

### 12.2 시뮬 결과
- **D11 위 트레이 위치 허용 범위** (`scripts/d11tol.py`, n, 76 mm, 깊게 잡기 + 낮게 꺼내기): 레일 방향(base +y) −3 ~ +1 cm (+2 cm는 팔이 안 닿음), 포크 방향 ±3 cm, yaw −3 ~ +10° (−6°는 운반 중 base에 닿음).
- **관절 오차** (`scripts/perturb_run.py`): 0.5°는 모두 통과. 1°는 D11 쪽(꺼내기: 로봇 받침/포크, 운반: base)과 책상 → 버거 운반(base)에서 닿음.
- **책상 자리** (`scripts/spotscan.py`, 42곳): 17~28 cm에서 27곳 성공, 자리에 따라 거의 차이 없음 (여유, 5 mm 안 자세, 자세 전환 23~25 s 모두 같음). base 근처 문제는 자리가 아니라 **D11에서 당겨 오는 거리** `--relay-pull-r` 0.16 때문: 0.18 이상이면 D11 쪽도 5 mm, 5 mm 안 자세 0개 (기본값 변경은 아직 안 함, 0.20 제안).
- **계획 시간** (같은 배치, `scripts/profile_run.py`): 테스트 e 60 → 9 s, 선반 e 25 → 4.5 s, 선반 n 10+29 → 6+6 s, 선반 j 49 → 12 s.
- 버거 박스 중심 305 / 310 mm에서 e → r 연속 성공 (295 mm에 놓고 집음), 313 mm는 바로 ERROR.

### 12.3 실물 결과 (테스트 트레이, e / r)
- 성공: r 15:50, e 16:01, e 17:06, r 17:09 (버거 마커 (239, −147) ~ (278, −106) mm, 박스 yaw −78 ~ −113°).
- 계획: e 79 → **12 s**, r 77 → **24 s** (IK 개선 후).
- 관절: 끝 자세 오차 0.5° 이내 (보정 후). 움직이는 중 계획보다 약 0.12 s 늦음 (2~4°), **경로 이탈은 트레이를 든 구간 0.8~1.65°** (손끝 약 4~8 mm, 여유 5 mm와 비슷) → 속도 낮추기 넣음.
- 잡기 간격 3.0 mm (손잡이 2~3 mm 두께, 정상).
- 실패 사례: 버거 30.5 / 31.1 cm → 계획 몇 분 후 실패 (지금은 거리 검사로 바로 ERROR). r을 박스가 빈 상태로 실행 → "nothing found" 후 홈 (정상).

### 12.4 다음에 할 일
1. 버거 도킹 오차 확인 (사용자, 11-25): 위 범위(거리 230~310 mm 놓기 / 230~300 mm 집기) 안에 드는지.
2. 실물에서 `--relay-held-speed 0.25` 효과: `track_log.csv` socket 줄의 off_norm 비교.
3. 실물에서 빈손 이동 높이 (책상 위 4 cm, 먼저 올리기)와 홈 복귀 처짐 확인.
4. `--relay-pull-r 0.20` 기본값 여부 결정 (n / j의 D11 쪽 여유 3 → 5 mm).
5. 12.1의 커밋 안 한 수정 커밋.
6. 남은 제안: 경로 IK 간격 5 mm → 1 cm (계획 약 절반), 테스트 트레이에서 되는 툴 방향 먼저 시도, 드라이버 lookahead 튜닝, e에서 버거 마커도 가까이서 다시 재기.

### 12.5 시뮬 / 실물 명령
```bash
# 시뮬 (선반 없음, 테스트 트레이): layout_noshelf.yaml = layout.yaml에서 with_shelf: false
ros2 launch mycobot_calibration handeye.launch.py sim:=true layout:=false scene_args:="--layout $HOME/.ros/layout_noshelf.yaml"
ros2 run mycobot_tasks layout --file $HOME/.ros/layout_noshelf.yaml --tray-height 0.072 --scene-args="--layout $HOME/.ros/layout_noshelf.yaml"
ros2 run mycobot_tasks tray_handle_pick --tray-type test --layout-poses --scene-args="--layout $HOME/.ros/layout_noshelf.yaml" --go   # 창에서 e / r
# 실물
ros2 launch mycobot_calibration handeye.launch.py scene_args:="--layout $HOME/.ros/layout_noshelf.yaml"
ros2 run mycobot_tasks tray_handle_pick --tray-type test --go --real          # 처음엔 --auto 없이 단계 모드
ros2 run mycobot_tasks tray_handle_pick --tray-type test --go --real --do bot --auto   # 버거 위치만 재기
# 분석 (notes/.../scripts): profile_run.py (계획 시간 분해), perturb_run.py (관절 오차), d11tol.py, spotscan.py
```
