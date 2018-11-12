#!/usr/bin/env python3
# Python 3.6

import hlt
import time
import ptvsd
import random
import logging
import numpy as np
from hlt import constants
from hlt.positionals import Direction

class MyBot:

    def __init__(self):
        self.game = hlt.Game()

        #Initialize input layers
        self.halite_locations = np.zeros((64,64))
        self.current_unit = np.zeros((64,64))
        self.unit_score = np.zeros((64,64))
        self.my_units = np.zeros((64,64))
        self.shipyards = np.zeros((64,64))
        self.dropoffs = np.zeros((64,64))
        self.score = np.zeros((64,64))

        #TODO: Generalize for 2 and 4 players
        self.enemies = np.zeros((64,64))
        self.enemy_halite = np.zeros((64,64))
        self.enemy_score = np.zeros((64,64))
        
        self.game.ready("MyPythonBot")
        logging.info("Successfully created bot! My Player ID is {}.".format(self.game.my_id))


    def run(self):
        ptvsd.enable_attach(address=('localhost', 5678))
        ptvsd.wait_for_attach()

        while True:
            # This loop handles each turn of the game. The game object changes every turn, and you refresh that state by
            #   running update_frame().
            self.game.update_frame()
            # You extract player metadata and the updated map metadata here for convenience.
            me = self.game.me
            game_map = self.game.game_map

            for moreCells in game_map._cells:
                for cell in moreCells:
                    if cell.ship is not None:
                        logging.info("Owner" + str(cell.ship.owner) + " Halite: " +  str(cell.ship.halite_amount))

            # A command queue holds all the commands you will run this turn. You build this list up and submit it at the
            #   end of the turn.
            command_queue = []

            for ship in me.get_ships():
                # For each of your ships, move randomly if the ship is on a low halite location or the ship is full.
                #   Else, collect halite.
                logging.info(ship.position)
                if game_map[ship.position].halite_amount < constants.MAX_HALITE / 10 or ship.is_full:
                    command_queue.append(
                        ship.move(
                            random.choice([ Direction.North, Direction.South, Direction.East, Direction.West ])))
                else:
                    command_queue.append(ship.stay_still())

            # If the game is in the first 200 turns and you have enough halite, spawn a ship.
            # Don't spawn a ship if you currently have a ship at port, though - the ships will collide.
            if self.game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
                command_queue.append(me.shipyard.spawn())

            # Send your moves back to the game environment, ending this turn.
            self.game.end_turn(command_queue)


if __name__ == '__main__':
    bot = MyBot()
    bot.run()
