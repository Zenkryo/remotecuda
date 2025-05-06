#ifndef __CLIENT_H__
#define __CLIENT_H__
#include <string.h>
#include "rpc/rpc_core.h"
using namespace rpc;

RpcConn *rpc_get_conn();
void rpc_release_conn(RpcConn *conn);

#endif
