#ifndef COMMAND_HPP
#define COMMAND_HPP

#include "Entity.hpp"
#include "Map.hpp"
#include "Player.hpp"
#include <memory>

namespace hlt {

class CommandTransaction;

/** Abstract type of commands issued by the user. */
class Command {
public:
    /** The command names. */
    enum Name : char {
        Move = 'm',
        Spawn = 'g',
        Construct = 'c'
    };


    /**
     * Convert to bot serial format.
     * @return The serialized command.
     */
    virtual std::string to_bot_serial() const = 0;

    /**
     * Add the command to a transaction.
     * @param player The player executing the command.
     * @param transaction The command transaction to act on.
     */
    virtual void add_to_transaction(Player &player, CommandTransaction &transaction) const = 0;

    /** Virtual destructor. */
    virtual ~Command() = default;
};

/** Statically polymorphic mixin for commands that may be added to a transaction. */
template<class T>
class TransactableCommand : public Command {
public:
    /**
     * Add the command to a transaction.
     * @param player The player executing the command.
     * @param transaction The command transaction to act on.
     */
    void add_to_transaction(Player &player, CommandTransaction &transaction) const final;

    /** Virtual destructor. */
    ~TransactableCommand() override = default;
};

/**
 * Read a Command from bot serial format.
 * @param istream The input stream.
 * @param[out] command The command to read.
 * @return The input stream.
 */
std::istream &operator>>(std::istream &istream, std::unique_ptr<Command> &command);

/**
 * Write a Command to bot serial format.
 * @param ostream The output stream.
 * @param[out] command The command to write.
 * @return The output stream.
 */
std::ostream &operator<<(std::ostream &ostream, std::unique_ptr<Command> &command);

/** Command for moving an entity in a direction. */
class MoveCommand final : public TransactableCommand<MoveCommand> {
public:
    const Entity::id_type entity; /**< The location of the entity. */
    const Direction direction;    /**< The direction in which to move. */

    /**
     * Convert to bot serial format.
     * @return The serialized command.
     */
    std::string to_bot_serial() const override;

    /**
     * Create MoveCommand from entity and direction.
     * @param entity The location of the entity.
     * @param direction The direction.
     */
    MoveCommand(const Entity::id_type &entity, Direction direction) : entity(entity), direction(direction) {}
};

/** Command for spawning an entity. */
class SpawnCommand final : public TransactableCommand<SpawnCommand> {
public:

    /**
     * Convert to bot serial format.
     * @return The serialized command.
     */
    std::string to_bot_serial() const override;

    /**
     * Create SpawnCommand from energy.
     * @param energy The energy to use.
     */
    explicit SpawnCommand() {}
};

/** Command to construct a drop zone. */
class ConstructCommand final : public TransactableCommand<ConstructCommand> {
public:
    /** The entity to use to construct. */
    const Entity::id_type entity;

    /**
     * Convert to bot serial format.
     * @return The serialized command.
     */
    std::string to_bot_serial() const override;

    /**
     * Construct ConstructCommand from entity.
     * @param entity The entity to convert.
     */
    explicit ConstructCommand(const Entity::id_type &entity) : entity(entity) {}
};

}

#endif // COMMAND_HPP
