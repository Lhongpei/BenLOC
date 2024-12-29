def print_logo():
    print("""
                                                        ,--,                             
                                        ,---.'|       ,----..               
            ,---,.                      |   | :      /   /   \    ,----..   
        ,'  .'  \                     :   : |     /   .     :  /   /   \  
        ,---.' .' |               ,---, |   ' :    .   /   ;.  \|   :     : 
        |   |  |: |           ,-+-. /  |;   ; '   .   ;   /  ` ;.   |  ;. / 
        :   :  :  /   ,---.  ,--.'|'   |'   | |__ ;   |  ; \ ; |.   ; /--`  
        :   |    ;   /     \|   |  ,"' ||   | :.'||   :  | ; | ';   | ;     
        |   :     \ /    /  |   | /  | |'   :    ;.   |  ' ' ' :|   : |     
        |   |   . |.    ' / |   | |  | ||   |  ./ '   ;  \; /  |.   | '___  
        '   :  '; |'   ;   /|   | |  |/ ;   : ;    \   \  ',  / '   ; : .'| 
        |   |  | ; '   |  / |   | |--'  |   ,/      ;   :    /  '   | '/  : 
        |   :   /  |   :    |   |/      '---'        \   \ .'   |   :    /  
        |   | ,'    \   \  /'---'                     `---`      \   \ .'   
        `----'       `----'                                       `---`     
                """)
    print("Welcome to BenLOC, a benchmark of Learning to MIP Optimizer Configuration.")
    print("This library is developed by the team at SUFE & Cardinal Optimizer.")
    print("If you have any questions, please contact us at ishongpeili@gmail.com.")

def log_init(func):
    def wrapper(*args, **kwargs):
        print_logo()
        return func(*args, **kwargs)
    return wrapper