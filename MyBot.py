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
from hlt.entity import Shipyard, Dropoff

class MyBot:

    def __init__(self):
        self.game = hlt.Game()

        self.height = self.game.game_map.height
        self.width = self.game.game_map.width

        #Initialize settings for different board sizes
        if self.height == 64:
            self.num_steps = 500
            self.offset = 0
        elif self.height == 50:
            self.num_steps = 475
            self.offset = 7
        elif self.height == 40:
            self.num_steps = 425
            self.offset = 12
        else:
            self.num_steps = 400
            self.offset = 16
        
        ptvsd.enable_attach(address=('localhost', 5678))
        ptvsd.wait_for_attach()

        #Board info
        self.halite_locations = np.zeros((64,64))
        self.steps_remaining = np.zeros((64,64))
        #My global info
        self.my_ships = np.zeros((64,64))
        self.my_halite = np.zeros((64,64))
        self.dropoffs = np.zeros((64,64))
        self.score = np.zeros((64,64))
        #My current unit info
        self.unit_halite = np.zeros((64,64))
        self.current_unit = np.zeros((64,64))
        #Enemy global info
        #TODO: Generalize for 2 and 4 players
        self.enemy_ships = np.zeros((64,64))
        self.enemy_halite = np.zeros((64,64))
        self.enemy_score = np.zeros((64,64))
        
        self.game.ready("MyPythonBot")
        logging.info("Successfully created bot! My Player ID is {}.".format(self.game.my_id))

    def buildInputs(self, cells):

        for row in cells:
            for cell in row:
                x = self.offset + cell.position.x
                y = self.offset + cell.position.y
                self.halite_locations[x][y] = cell.halite_amount

                #Note: both dropoffs and shipyards count as "Dropoffs"
                if cell.has_structure:
                    if cell.structure.owner == self.game.my_id:
                        self.dropoffs[x][y] = 1
                    else:
                        #TODO: Do we care about enemy dropoffs?
                        pass

                if cell.is_occupied:
                    if cell.ship.owner == self.game.my_id:
                        self.my_ships[x][y] = 1
                        self.my_halite[x][y] = cell.ship.halite_amount
                    else:
                        self.enemy_ships[x][y] = 2
                        self.enemy_halite[x][y] = cell.ship.halite_amount

        self.score = np.full((64,64), self.game.me.halite_amount)
        self.steps_remaining = np.full((64,64), self.num_steps - self.game.turn_number + 1)

        for id, player in self.game.players.items():
            if id != self.game.my_id:
                self.enemy_score = np.full((64,64), player.halite_amount)

    def run(self):

        while True:
            # This loop handles each turn of the game. The game object changes every turn, and you refresh that state by
            #   running update_frame().
            self.game.update_frame()
            # You extract player metadata and the updated map metadata here for convenience.
            me = self.game.me
            game_map = self.game.game_map

            self.buildInputs(game_map._cells)

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
