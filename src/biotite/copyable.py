# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["Copyable"]

import abc


class Copyable(metaclass=abc.ABCMeta):
    """
    Base class for all objects, that should be copyable.
    
    The public method `copy()` first creates a fresh instance of the
    class of the instance, that is copied via the `__copy_create__()`
    method. All variables, that could not be set via the constructor,
    are then copied via `__copy_fill__()`, starting with the method in
    the uppermost base class and ending with the class of the instance
    to be copied.
    
    This approach solves the problem of encapsulated variables in
    superclasses.
    """
    
    def copy(self):
        """
        Copy the object.
        
        Returns
        -------
        copy
            A copy of this object.
        """
        clone = self.__copy_create__()
        self.__copy_fill__(clone)
        return clone
    
    def __copy_create__(self):
        """
        Instantiate a new object of this class.
        
        Only the constructor should be called in this method.
        All further attributes, that need to be copied are handled
        in `__copy_fill__()`
        
        Do not call the `super()` method here.
        
        This method must be overridden, if the constructor takes
        parameters.
        
        Returns
        -------
        copy
            A freshly instantiated copy of *self*.
        """
        return type(self)()
    
    def __copy_fill__(self, clone):
        """
        Copy all necessary attributes to the new object.
        
        Always call the `super()` method as first statement.
        
        Parameters
        ----------
        clone
            The freshly instantiated copy of *self*.
        """
        pass