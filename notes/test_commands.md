# tray_handle_pick 테스트 명령 모음 (2026-11-26 기준, 커밋 2c0dee6)

동작 키: **n** 선반 D11 → 책상 → 버거 · **j** 버거 → 책상 → D11 · **e** 책상 → 버거 · **r** 버거 → 책상 ·
**d** 트레이 측정 · **s** 탐색 자세 · **c** RViz 책상 목표 지우기 · **a** 자동/단계 · **x / SPACE** 정지 · **q** 종료

---

## 0. 먼저 알아둘 것

| 항목 | 내용 |
|---|---|
| 트레이 종류 | 지금 실물은 **테스트 트레이** → 항상 `--tray-type test` (16×10 cm, 손잡이는 10 cm 변 방향). 선반 트레이(15×10)는 옵션 없이(기본값) |
| 장면 기본값 | 책상 = layout 값(edge_y −0.506), **선반 없음**, 베이스 뒤 Pi 포트 케이블(랜/파워) 포함 |
| n / j | **실물에서 아직 금지.** 시뮬에서 `--relay-pull-r 0.20` + 선반 트레이 상자 방향 0°로 다시 확인한 뒤 실물 |
| 코드 바꾼 뒤 | `tray_handle_pick`은 다시 켜야 반영. 드라이버 / RViz 설정은 launch를 다시 켜야 반영 |
| 시뮬과 실물 동시에 | 같은 PC에서 둘 다 돌리면 섞임 → 시뮬 쪽 **모든 터미널**에서 `export ROS_DOMAIN_ID=77` 먼저 |
| 빌드 | `cd ~/final_project_ws && colcon build --packages-select mycobot_tasks && source install/setup.bash` |

---

## 1. 실물 (실제 로봇)

### 1-1. launch
```bash
# 선반 없음 (e / r 기본)
ros2 launch mycobot_calibration handeye.launch.py connection:=socket ip:=10.42.0.188 camera_device:=2

# 선반 장면 포함 (n / j 용: ~/.ros/layout.yaml 의 선반 + 배치)
ros2 launch mycobot_calibration handeye.launch.py connection:=socket ip:=10.42.0.188 camera_device:=2 \
  scene_args:=--layout

# 선반 없이 layout 파일 배치만 (책상 위치는 기본값과 같음, 고정 버거 모델 포함)
ros2 launch mycobot_calibration handeye.launch.py connection:=socket ip:=10.42.0.188 camera_device:=2 \
  scene_args:="--layout $HOME/.ros/layout_noshelf.yaml"
```

### 1-2. e / r (테스트 트레이, 선반 없음)
```bash
ros2 run mycobot_tasks tray_handle_pick --go --real --tray-type test            # 단계 모드 (Enter로 진행)
ros2 run mycobot_tasks tray_handle_pick --go --real --tray-type test --auto     # 자동
```
- 책상 자리: 기본 (−200, −140) → 없으면 (−120, −220)을 봄. r은 (−200, −140)에 놓음.
- **RViz에서 책상 위치/방향 정하기**: 툴바 **2D Goal Pose** → 책상 위 클릭 후 끌기
  (점 = 손잡이 중심, 화살표 = L→R = 마커 10→11). r은 거기에 놓고 e는 거기서 찾음.
  - 초록 호 두 개(반지름 19 / 29 cm) **사이**만 받음. 밖이면 무시.
  - 27 cm 넘으면 "낮게 들고(3 cm) 베이스 쪽으로 4~8 cm 당긴 뒤 올림"으로 동작.
  - 창에서 **c** = 목표 지우기 (기본 자리로).
  - 케이블(x −7.8 cm 선, −y로 책상 가로지름) 위에는 r이 놓지 않음 (e는 케이블 위에서도 집음).
- 처음 시작할 때 바로 한 동작: `--do deskbot` (e) / `--do botdesk` (r).

### 1-3. n / j (선반 트레이, 선반 장면) — 시뮬 확인 후에만
```bash
# launch는 1-1의 "선반 장면 포함"
ros2 run mycobot_tasks tray_handle_pick --go --real --scene-args=--layout --relay-pull-r 0.20
# 테스트 트레이로 선반을 쓸 때만 --tray-type test 추가
```
- `--relay-pull-r 0.20`: 선반에서 꺼낸 트레이를 베이스 쪽으로 당기는 반지름. 기본 0.16은 베이스까지 3~4 mm라 위험.

### 1-4. 드라이버 설정 (실행 중 변경, 다음 이동부터 적용)
```bash
ros2 param set /mycobot_driver ff_along_path true     # 경로 따라 중력 보정 (기본 켬)
ros2 param set /mycobot_driver lookahead 0.1          # 0..0.5 s (0.15 시험: 차이 없음)
ros2 param set /mycobot_driver loop_rate 20.0         # 5..60 Hz (실제 소켓 한계 ~17.6 / 21 Hz)
ros2 param set /mycobot_driver stream_speed 80        # 1..100
```

---

## 2. 시뮬레이션 (fake 드라이버)

실물 launch가 켜져 있으면 시뮬 쪽 터미널마다 먼저:
```bash
export ROS_DOMAIN_ID=77
```

### 2-1. launch
```bash
# 선반 있음 (layout 노드 자동: RViz에서 책상/버거/선반/D11 끌어 배치, ~/.ros/layout.yaml)
ros2 launch mycobot_calibration handeye.launch.py sim:=true

# 선반 없음 (실물 기본 장면과 같음)
ros2 launch mycobot_calibration handeye.launch.py sim:=true layout:=false

# 선반 없음 + layout 파일의 버거 위치 고정 모델 (README 12.5 방식)
ros2 launch mycobot_calibration handeye.launch.py sim:=true layout:=false \
  scene_args:="--layout $HOME/.ros/layout_noshelf.yaml"
# 이 경우 RViz에서 버거/트레이를 끌고 싶으면 (선반 없는 파일로)
ros2 run mycobot_tasks layout --file $HOME/.ros/layout_noshelf.yaml --tray-height 0.072 \
  --scene-args="--layout $HOME/.ros/layout_noshelf.yaml"
```

### 2-2. e / r (시뮬)
```bash
# 창에서 e / r (트레이·버거 위치 = layout 파일, 카메라 없이)
ros2 run mycobot_tasks tray_handle_pick --tray-type test --layout-poses --go

# 계획만 (움직이지 않음, 바로 종료)
ros2 run mycobot_tasks tray_handle_pick --tray-type test --layout-poses --auto --no-show --do deskbot   # e
ros2 run mycobot_tasks tray_handle_pick --tray-type test --layout-poses --auto --no-show --do botdesk   # r

# 책상 위 트레이 위치/방향을 직접 (x, y [m], L->R [deg])
ros2 run mycobot_tasks tray_handle_pick --tray-type test --layout-poses --auto --no-show --do deskbot \
  --relay-desk=-0.217,-0.187,-174
```
- `--scene-args=--layout`을 붙이면 선반이 장면에 들어감 (e / r 경로가 막힐 수 있음).

### 2-3. n / j (시뮬, 선반 장면)
```bash
ros2 launch mycobot_calibration handeye.launch.py sim:=true            # 선반 있는 launch
ros2 run mycobot_tasks tray_handle_pick --layout-poses --scene-args=--layout --relay-pull-r 0.20 --go
# 계획만
ros2 run mycobot_tasks tray_handle_pick --layout-poses --scene-args=--layout --relay-pull-r 0.20 \
  --auto --no-show --do d11bot      # n   (j = --do botd11)
```

---

## 3. 결과 확인

```bash
# 놓은 뒤 다시 잰 오차 (실물 e / r / n 마다 1줄; 1 cm / 5° 넘으면 WARNING)
column -s, -t < ~/.ros/place_check.csv | tail -20

# 경로 이탈 (트레이를 든 동작, 동작 이름별)
cd ~/final_project_ws/notes/reach_study_2026-11-20/scripts
python3 track_summary.py --since 14:00                 # 요약
python3 track_summary.py --since 14:00 --rows          # 동작마다
python3 track_summary.py --since 14:00 --by ff         # 드라이버 설정별 비교 (ff / lookahead / rate / stream / speed)

# 실행 로그 (말한 내용 전부)
tail -100 ~/.ros/tray_handle_pick.log
```

## 4. 분석 스크립트 (시뮬, `ROS_DOMAIN_ID=77`, 선반 없는 launch)
```bash
cd ~/final_project_ws
# 트레이를 든 경로마다 베이스(base / link1 / robot_mount)까지 최소 간격
python3 notes/reach_study_2026-11-20/scripts/basegap_run.py --tray-type test --layout-poses --auto --no-show --do deskbot
# 책상 위치 x 방향별 e(또는 r) 계획 지도 (45 s 넘으면 실패로 침)
python3 notes/reach_study_2026-11-20/scripts/deskcover.py /tmp/cover.csv --do deskbot \
  --step 0.05 --x=-0.35,0.05 --y=-0.35,0.10 --r 0.15,0.33 --yaws 0,60,120 --timeout 45
```

---

## 5. 자주 나온 문제

| 증상 | 원인 / 대처 |
|---|---|
| `relay plan ... no way` + `no IK` / `IK branch jump` (책상) | 트레이가 베이스에서 **29 cm 넘게** 멀다 → 26 cm 안으로 옮기기 (RViz 초록 호 안쪽) |
| 놓은 뒤 4~8 cm / 수십 도 밀림 | 선(책상 케이블 또는 **팔의 카메라 선**)에 끌림 → 선 정리, 케이블 위에 놓지 않기 |
| `nothing found in the TurtleBot area (... not reachable)` | 취소 뒤 그리퍼에 트레이 모델이 남음 → 아래 명령으로 제거 후 다시 |
| `Burger heading marker ID 28 not found` | 라이다 뒤 ID 28(22.5 mm)이 안 보임 → 마커 가림/방향 확인. 못 찾으면 r은 계획 안 함 |
| `unknown option(s): ... (did you mean ...?)` | 옵션 오타 (예: `--relay-held-spped`) |

```bash
# 그리퍼에 남은 트레이 모델 제거 (로봇은 안 움직임, 그리퍼가 비어 있을 때만)
ros2 service call /apply_planning_scene moveit_msgs/srv/ApplyPlanningScene \
  "{scene: {is_diff: true, robot_state: {is_diff: true, attached_collision_objects: [{link_name: gripper_tcp, object: {id: relay_tray, operation: 1}}]}}}"
```

## 6. 오늘(11-26) 실물 기준 값
- 놓은 위치 오차: r(책상) 평균 2.7 mm / 최대 4.1 mm, e(버거 상자) 평균 4.8 mm / 최대 8.4 mm, 각도 1.5° 이하 (선에 끌린 2건 제외)
- 트레이를 든 경로 이탈: `ff_along_path` 켠 뒤 평균 0.6° / 최대 0.9° (켜기 전 1.0° / 1.7°)
- 속도 올리기(빈손 0.5 rad/s, MoveIt 0.8)는 시험 후 되돌림 → 선반 n/j가 된 다음에
