use crate::utils::Timer;
use clap::{crate_name, crate_version};
use log::info;

const LOG_WIDTH: usize = 80;

pub fn write_header() {
    info!("{: ^LOG_WIDTH$}", "-----------------");
    info!("{: ^LOG_WIDTH$}", crate_name!().to_uppercase());
    info!("{: ^LOG_WIDTH$}", "-----------------");
    info!("{: ^LOG_WIDTH$}", format!("version: {}", crate_version!()));
    info!("{: ^LOG_WIDTH$}", "");
    info!("{: ^LOG_WIDTH$}", format!("{::^55}", ""));
    info!(
        "{: ^80}",
        "::                   Roland Mitric                   ::"
    );
    info!(
        "{: ^80}",
        "::  Institute of Physical and Theoretical Chemistry  ::"
    );
    info!(
        "{: ^80}",
        "::              University of Wuerzburg              ::"
    );
    info!(
        "{: ^80}",
        "::::::...................................................::::::"
    );
    info!(
        "{: ^80}",
        ":: Contributors:                                             ::"
    );
    info!(
        "{: ^80}",
        ":: --------                                                  ::"
    );
    info!(
        "{: ^80}",
        ":: Joscha Hoche             <joscha.hoche@uni-wuerzburg.de>  ::"
    );
    info!(
        "{: ^80}",
        ":: Richard Einsele       <richard.einsele@uni-wuerzburg.de>  ::"
    );
    info!(
        "{: ^80}",
        ":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
    );
    info!("{: ^80}", "");
}

pub fn write_footer(timer: Timer) {
    info!("{}", timer);
    info!("{: ^80}", "");
    info!("{: ^80}", "::::::::::::::::::::::::::::::::::::::");
    info!(
        "{: ^80}",
        format!(
            "::   Thank you for using {}    ::",
            crate_name!().to_uppercase()
        )
    );
    info!("{: ^80}", "::::::::::::::::::::::::::::::::::::::");
    info!("{: ^80}", "");
}
