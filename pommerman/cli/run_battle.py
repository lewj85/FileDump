"""Run a battle among agents.
Call this with a config, a game, and a list of agents. The script will start separate threads to operate the agents
and then report back the result.
An example with all four test agents running ffa:
python run_battle.py --agents=test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent --config=ffa_v0
An example with one player, two random agents, and one test agent:
python run_battle.py --agents=player::arrows,test::agents.SimpleAgent,random::null,random::null --config=ffa_v0
An example with a docker agent:
python run_battle.py --agents=player::arrows,docker::pommerman/test-agent,random::null,random::null --config=ffa_v0
"""
import atexit
import os
import random
import time

import argparse
import numpy as np

from .. import helpers
from .. import make
import json
from .. import utility

def featurize(obs):
    board = obs["board"].reshape(-1).astype(np.float32)
    bomb_blast_strength = obs["bomb_blast_strength"].reshape(-1) \
                                                    .astype(np.float32)
    bomb_life = obs["bomb_life"].reshape(-1).astype(np.float32)
    position = utility.make_np_float(obs["position"])
    ammo = utility.make_np_float([obs["ammo"]])
    blast_strength = utility.make_np_float([obs["blast_strength"]])
    can_kick = utility.make_np_float([obs["can_kick"]])

    teammate = utility.make_np_float([obs["teammate"].value])
    enemies = utility.make_np_float([e.value for e in obs["enemies"]])
    return np.concatenate((
        board, bomb_blast_strength, bomb_life, position, ammo,
        blast_strength, can_kick, teammate, enemies))


def save_stuff(obs, all_actions, steps, i, done, reward):
    agent_state = featurize(obs[0])
    p = agent_state.astype(np.int8)
    q = p.flatten().tolist()
    # remove 9,8,5,3,2,1 from end
    q2 = q[0:-4]
    q2.pop(-5)
    q2.pop(-4)
    # find pos of you, teammate, enemy1, enemy2

    myx, myy = obs[0]["position"]
    enemy1x, enemy1y = obs[1]["position"]
    teammatex, teammatey = obs[2]["position"]
    enemy2x, enemy2y = obs[3]["position"]

    # manhattan distance to teammate
    q2.append(abs(myx - teammatex) + abs(myy - teammatey))
    q2.append(abs(myx - enemy1x) + abs(myy - enemy1y))
    q2.append(abs(myx - enemy2x) + abs(myy - enemy2y))
    q2.append(abs(teammatex - enemy1x) + abs(teammatey - enemy1y))
    q2.append(abs(teammatex - enemy2x) + abs(teammatey - enemy2y))

    # hot-one encode first 169 values then tack q2[169:] onto the end of q3:
    #   [empty,wall,
    q3 = []
    for ind in range(169):
        # no fog(5), agentdummy(10), both enemies get same value
        arr = [0] * 12
        val = q2[ind]
        # Item in constants.py
        if val in [0, 1, 2, 3, 4]:
            arr[val] = 1
        elif val in [6, 7, 8, 9]:
            arr[val - 1] = 1
        elif val in [11, 12, 13]:
            arr[val - 2] = 1
        elif val == 14:
            arr[10] = 1
        q3.append(arr)
    q4 = [item for sublist in q3 for item in sublist]
    q5 = [q4, q2[169:]]
    q6 = [int(item) for sublist in q5 for item in sublist]

    r = np.asarray(all_actions, dtype=np.int8)
    s = r.tolist()
    d = {
        "state": q6,
        "actions": s
    }
    e = json.dumps(d)

    with open(f'./jesse/{i}.json', 'a+') as f:
        if steps == 1:
            f.write('{"states": [')
        f.write(e)
        if done:
            f.write('], "reward":' + str(reward[0]) + '}')
        else:
            f.write(',')


def run(args, num_times=1, seed=None):
    config = args.config
    record_pngs_dir = args.record_pngs_dir
    record_json_dir = args.record_json_dir
    agent_env_vars = args.agent_env_vars
    game_state_file = args.game_state_file

    # TODO: After https://github.com/MultiAgentLearning/playground/pull/40
    #       this is still missing the docker_env_dict parsing for the agents.
    agents = [
        helpers.make_agent_from_string(agent_string, agent_id+1000)
        for agent_id, agent_string in enumerate(args.agents.split(','))
    ]

    env = make(config, agents, game_state_file)

    if args.record_pngs_dir:
        assert not os.path.isdir(args.record_pngs_dir)
        os.makedirs(args.record_pngs_dir)
    if args.record_json_dir:
        assert not os.path.isdir(args.record_json_dir)
        os.makedirs(args.record_json_dir)

    def _run(seed, record_pngs_dir=None, record_json_dir=None, i=0):
        env.seed(seed)
        print("Starting the Game.")
        obs = env.reset()
        steps = 0
        done = False
        while not done:
            steps += 1
            if args.render:
                env.render(record_pngs_dir=args.record_pngs_dir,
                           record_json_dir=args.record_json_dir)
            actions = env.act(obs)
            obs, reward, done, info = env.step(actions)
            save_stuff(obs, actions, steps, i, done, reward)

        for agent in agents:
            agent.episode_end(reward[agent.agent_id])
        
        print("Final Result: ", info)
        if args.render:
            time.sleep(5)
            env.render(record_pngs_dir=args.record_pngs_dir,
                       record_json_dir=args.record_json_dir, close=True)
        return info

    infos = []
    times = []
    for i in range(num_times):
        start = time.time()
        if seed is None:
            seed = random.randint(0, 1e6)
        np.random.seed(seed)
        random.seed(seed)

        record_pngs_dir_ = record_pngs_dir + '/%d' % (i+1) \
                           if record_pngs_dir else None
        record_json_dir_ = record_json_dir + '/%d' % (i+1) \
                           if record_json_dir else None
        infos.append(_run(seed, record_pngs_dir_, record_json_dir_, i))

        times.append(time.time() - start)
        print("Game Time: ", times[-1])

    atexit.register(env.close)
    return infos


def main():
    simple_agent = 'test::agents.SimpleAgent'
    player_agent = 'player::arrows'
    docker_agent = 'docker::pommerman/simple-agent'
    
    parser = argparse.ArgumentParser(description='Playground Flags.')
    parser.add_argument('--game',
                        default='pommerman',
                        help='Game to choose.')
    parser.add_argument('--config',
                        default='PommeFFA-v0',
                        help='Configuration to execute. See env_ids in '
                        'configs.py for options.')
    parser.add_argument('--agents',
                        default=','.join([simple_agent]*4),
                        # default=','.join([player_agent] + [simple_agent]*3]),
                        # default=','.join([docker_agent] + [simple_agent]*3]),
                        help='Comma delineated list of agent types and docker '
                        'locations to run the agents.')
    parser.add_argument('--agent_env_vars',
                        help='Comma delineated list of agent environment vars '
                        'to pass to Docker. This is only for the Docker Agent.'
                        " An example is '0:foo=bar:baz=lar,3:foo=lam', which "
                        'would send two arguments to Docker Agent 0 and one '
                        'to Docker Agent 3.',
                        default="")
    parser.add_argument('--record_pngs_dir',
                        default=None,
                        help='Directory to record the PNGs of the game. '
                        "Doesn't record if None.")
    parser.add_argument('--record_json_dir',
                        default=None,
                        help='Directory to record the JSON representations of '
                        "the game. Doesn't record if None.")
    parser.add_argument('--render',
                        default=True,
                        help="Whether to render or not. Defaults to True.")
    parser.add_argument('--game_state_file',
                        default=None,
                        help="File from which to load game state.")
    args = parser.parse_args()
    run(args, 2)


if __name__ == "__main__":
    main()