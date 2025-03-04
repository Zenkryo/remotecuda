#ifndef SERVER_H
#define SERVER_H

#include <cstdint>

typedef int (*RequestHandler)(void *);

RequestHandler get_handler(const uint32_t funcId);

#endif // SERVER_H