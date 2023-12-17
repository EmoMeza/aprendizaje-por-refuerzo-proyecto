# Importaciones necesarias
import asyncio
import numpy as np
import os
import sys

from gym.spaces import Space, Box
from gym.utils.env_checker import check_env
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tabulate import tabulate

import tensorflow as tf
import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.data import GenData
from poke_env.player import (
    Gen8EnvSinglePlayer,
    RandomPlayer,
    MaxBasePowerPlayer,
    SimpleHeuristicsPlayer,
    wrap_for_old_gym_api,
)

my_team = """
    Corviknight @ Leftovers  
    Ability: Mirror Armor  
    EVs: 252 HP / 4 Atk / 252 SpD  
    Careful Nature  
    - Roost  
    - Iron Defense  
    - Body Press  
    - Brave Bird  

    Urshifu-Rapid-Strike @ Choice Scarf  
    Ability: Unseen Fist  
    EVs: 252 Atk / 4 SpD / 252 Spe  
    Jolly Nature  
    - U-turn  
    - Aqua Jet  
    - Surging Strikes  
    - Close Combat  

    Tapu Koko @ Air Balloon  
    Ability: Electric Surge  
    EVs: 252 SpA / 4 SpD / 252 Spe  
    Timid Nature  
    IVs: 0 Atk  
    - Roost  
    - Protect  
    - Volt Switch  
    - Dazzling Gleam  

    Garchomp @ Yache Berry  
    Ability: Rough Skin  
    EVs: 252 Atk / 4 SpD / 252 Spe  
    Jolly Nature  
    - Protect  
    - Swords Dance  
    - Earthquake  
    - Dragon Claw  

    Rillaboom @ Assault Vest  
    Ability: Grassy Surge  
    EVs: 252 Atk / 4 SpD / 252 Spe  
    Jolly Nature  
    - Fake Out  
    - Grassy Glide  
    - U-turn  
    - Wood Hammer  

    Incineroar @ Iapapa Berry  
    Ability: Intimidate  
    EVs: 252 HP / 252 Atk / 4 SpD  
    Adamant Nature  
    - Parting Shot  
    - Fake Out  
    - Darkest Lariat  
    - Fire Punch  
""" 

# Creando la carpeta para guardar los pesos del modelo si no existe
weights_folder = "double_dqn_weights"
if not os.path.exists(weights_folder):
    os.makedirs(weights_folder)

# Clase del jugador utilizando aprendizaje por refuerzo
class SimpleRLPlayer(Gen8EnvSinglePlayer):
 # Calcula la recompensa basada en el estado del combate
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    # Transforma el estado del combate en una representación numérica
    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GenData.from_gen(8).type_chart
                )

        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6

        final_vector = np.concatenate(
            [moves_base_power, moves_dmg_multiplier, [fainted_mon_team, fainted_mon_opponent]]
        )
        return np.float32(final_vector)

    # Define el espacio de observaciones del agente
    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(np.array(low, dtype=np.float32), np.array(high, dtype=np.float32), dtype=np.float32)

# Función para alternar la actualización de los modelos
def update_target_model(model, target_model):
    target_model.set_weights(model.get_weights())

# Función principal del programa
async def main():
    # Configuración del entorno de prueba y entrenamiento
    opponent = SimpleHeuristicsPlayer(battle_format="gen8ou", team=my_team)
    test_env = SimpleRLPlayer(battle_format="gen8ou", start_challenging=True, opponent=opponent, team=my_team)
    check_env(test_env)
    test_env.close()

    train_env = SimpleRLPlayer(battle_format="gen8ou", opponent=opponent, start_challenging=True, team=my_team)
    train_env = wrap_for_old_gym_api(train_env)

    eval_env = SimpleRLPlayer(battle_format="gen8ou", opponent=opponent, start_challenging=True, team=my_team)
    eval_env = wrap_for_old_gym_api(eval_env)

    # Definición del modelo neuronal para el agente
    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape

    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=input_shape, kernel_initializer='he_uniform', use_bias=False, name="Initial"))
    model.add(Flatten())
    model.add(Dense(64, activation="elu", kernel_initializer='he_uniform', name="Middle"))
    model.add(Dense(n_action, activation="linear", kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01), name="Output"))

    # Crear un segundo modelo (target model) con la misma arquitectura
    target_model = keras.models.clone_model(model)

    # Configuración de la memoria y política del agente DQN
    memory = SequentialMemory(limit=500000, window_length=1)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr="eps", value_max=1.0, value_min=0.05, value_test=0.0, nb_steps=500000)
    dqn = DQNAgent(model=model, nb_actions=n_action, policy=policy, memory=memory, nb_steps_warmup=10000, gamma=0.5, target_model_update=1, delta_clip=0.01)
    dqn.compile(optimizer=Adam(learning_rate=0.0005), metrics=["mae"])

    # Carga de pesos si existen para ambos modelos
    weights_path = os.path.join(weights_folder, 'heur_500k_double_dqn_weights.h5f')
    if os.path.exists(weights_path):
        dqn.load_weights(weights_path)
        update_target_model(model, target_model)

    # Entrenamiento del agente con actualización alternada de los modelos
    for step in range(50000):
        dqn.fit(train_env, nb_steps=1, verbose=0)
        if step % 1000 == 0:  # Actualizar el target_model cada 1000 pasos
            update_target_model(model, target_model)

    # Guardado de los pesos después del entrenamiento
    dqn.save_weights(os.path.join(weights_folder, 'heur_500k_double_dqn_weights.h5f'), overwrite=True)
    train_env.close()
    print("Training done and saved.")

    # Evaluación del agente
    print("Results against random player:")
    dqn.test(eval_env, nb_episodes=1000, verbose=False, visualize=False)
    print(f"Double DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes")

    second_opponent = MaxBasePowerPlayer(battle_format="gen8ou", team=my_team)
    eval_env.reset_env(restart=True, opponent=second_opponent)
    print("Results against max base power player:")
    dqn.test(eval_env, nb_episodes=1000, verbose=False, visualize=False)
    print(f"Double DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes")

    eval_env.close()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
