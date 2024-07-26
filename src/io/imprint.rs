use crate::utils::Timer;
use clap::{crate_name, crate_version};
use log::{info, warn};

const LOG_WIDTH: usize = 80;

pub fn write_header() {
    warn!("{: ^LOG_WIDTH$}", "-----------------");
    warn!("{: ^LOG_WIDTH$}", crate_name!().to_uppercase());
    warn!("{: ^LOG_WIDTH$}", "-----------------");
    warn!("{: ^LOG_WIDTH$}", format!("version: {}", crate_version!()));
    warn!("{: ^LOG_WIDTH$}", "");
    warn!("{: ^LOG_WIDTH$}", format!("{::^55}", ""));
    warn!(
        "{: ^80}",
        "::                   Roland Mitric                   ::"
    );
    warn!(
        "{: ^80}",
        "::  Institute of Physical and Theoretical Chemistry  ::"
    );
    warn!(
        "{: ^80}",
        "::              University of Wuerzburg              ::"
    );
    warn!(
        "{: ^80}",
        "::::::...................................................::::::"
    );
    warn!(
        "{: ^80}",
        ":: Contributors:                                             ::"
    );
    warn!(
        "{: ^80}",
        ":: --------                                                  ::"
    );
    warn!(
        "{: ^80}",
        ":: Joscha Hoche             <joscha.hoche@uni-wuerzburg.de>  ::"
    );
    warn!(
        "{: ^80}",
        ":: Richard Einsele       <richard.einsele@uni-wuerzburg.de>  ::"
    );
    warn!(
        "{: ^80}",
        ":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
    );
    warn!("{: ^80}", "");
}

pub fn write_footer(timer: Timer) {
    warn!(
        "{:>68} {:>8.2} s",
        "total elapsed time:",
        timer.time.elapsed().as_secs_f32()
    );
    warn!("{: ^80}", "");
    warn!("{: ^80}", "::::::::::::::::::::::::::::::::::::::");
    warn!(
        "{: ^80}",
        format!(
            "::   Thank you for using {}    ::",
            crate_name!().to_uppercase()
        )
    );
    warn!("{: ^80}", "::::::::::::::::::::::::::::::::::::::");
    warn!("{: ^80}", "");
}
