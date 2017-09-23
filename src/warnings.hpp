#ifndef YKBCJNRTVSVXXTPEEPCBTODSCOGMLMTKXMDXDKNMXVHJUSRBJAIOASSOMOYKJHFITAQOXWLDG
#define YKBCJNRTVSVXXTPEEPCBTODSCOGMLMTKXMDXDKNMXVHJUSRBJAIOASSOMOYKJHFITAQOXWLDG

#ifdef __clang__

#define SUPPRESS_WARNINGS \
_Pragma("clang diagnostic push ") \
_Pragma("clang diagnostic ignored \"-Wshorten-64-to-32\"" ) \
_Pragma("clang diagnostic ignored \"-Wcast-align\"" ) \
_Pragma("clang diagnostic ignored \"-Wdouble-promotion\"" ) \
_Pragma("clang diagnostic ignored \"-Wreserved-id-macro\"" ) \
_Pragma("clang diagnostic ignored \"-Wdocumentation-unknown-command\"") \
_Pragma("clang diagnostic ignored \"-Wundef\"") \
_Pragma("clang diagnostic ignored \"-Wc++98-compat\"") \
_Pragma("clang diagnostic ignored \"-Wexit-time-destructors\"") \
_Pragma("clang diagnostic ignored \"-Wdocumentation-deprecated-sync\"") \
_Pragma("clang diagnostic ignored \"-Wdocumentation\"") \
_Pragma("clang diagnostic ignored \"-Wmissing-prototypes\"") \
_Pragma("clang diagnostic ignored \"-Wold-style-cast\"") \
_Pragma("clang diagnostic ignored \"-Wpadded\"") \
_Pragma("clang diagnostic ignored \"-Wc++98-compat-pedantic\"")
_Pragma("clang diagnostic ignored \"-Wzero-as-null-pointer-constant\"")


#define RESTORE_WARNINGS \
_Pragma( "clang diagnostic pop" )

#else

#define SUPPRESS_WARNINGS
#define RESTORE_WARNINGS

#endif

#endif//YKBCJNRTVSVXXTPEEPCBTODSCOGMLMTKXMDXDKNMXVHJUSRBJAIOASSOMOYKJHFITAQOXWLDG

