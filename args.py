from args_parser import get_parser, post_init_args, set_env

parser = get_parser()
if parser:
    args = parser.parse_args()
    post_init_args(args)
    set_env(args)
else:
    args = None
