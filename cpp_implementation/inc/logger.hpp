#ifndef LOGGER
#define LOGGER

#include <iostream>

class Logger
{
public:
    Logger() {}
    inline Logger(std::ostream &f, unsigned log_level, std::string &msg);
};

Logger::Logger(std::ostream &f, unsigned log_level, std::string &msg) {
    return;
}

#endif // LOGGER
