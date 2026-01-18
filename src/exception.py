import logging
import sys

def error_message_details(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()#in which file the error occured
    file_name = exc_tb.tb_frame.f_code.co_filename#line number where the error occured
    error_message = "Error Occured in python script name [{0}] and line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error))
    
    return error_message    

class CustomException(Exception):
    def __init__(self,error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail)

    def __str__(self):
        return self.error_message    
    